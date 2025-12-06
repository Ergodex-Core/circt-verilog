//===- Structure.cpp - Slang hierarchy conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/symbols/ClassSymbols.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/symbols/PortSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/ast/types/AllTypes.h"
#include "llvm/ADT/ScopeExit.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"

using namespace circt;
using namespace ImportVerilog;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static void guessNamespacePrefix(const slang::ast::Symbol &symbol,
                                 SmallString<64> &prefix) {
  if (symbol.kind != slang::ast::SymbolKind::Package)
    return;
  guessNamespacePrefix(symbol.getParentScope()->asSymbol(), prefix);
  if (!symbol.name.empty()) {
    prefix += symbol.name;
    prefix += "::";
  }
}

//===----------------------------------------------------------------------===//
// Base Visitor
//===----------------------------------------------------------------------===//

namespace {
/// Base visitor which ignores AST nodes that are handled by Slang's name
/// resolution and type checking.
struct BaseVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  BaseVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  // Skip semicolons.
  LogicalResult visit(const slang::ast::EmptyMemberSymbol &) {
    return success();
  }

  // Skip members that are implicitly imported from some other scope for the
  // sake of name resolution, such as enum variant names.
  LogicalResult visit(const slang::ast::TransparentMemberSymbol &) {
    return success();
  }

  // Skip typedefs.
  LogicalResult visit(const slang::ast::TypeAliasType &) { return success(); }
  LogicalResult visit(const slang::ast::ForwardingTypedefSymbol &) {
    return success();
  }

  // Skip imports. The AST already has its names resolved.
  LogicalResult visit(const slang::ast::ExplicitImportSymbol &) {
    return success();
  }
  LogicalResult visit(const slang::ast::WildcardImportSymbol &) {
    return success();
  }

  // Skip type parameters. The Slang AST is already monomorphized.
  LogicalResult visit(const slang::ast::TypeParameterSymbol &) {
    return success();
  }

  // Skip elaboration system tasks. These are reported directly by Slang.
  LogicalResult visit(const slang::ast::ElabSystemTaskSymbol &) {
    return success();
  }

  // Handle parameters.
  LogicalResult visit(const slang::ast::ParameterSymbol &param) {
    visitParameter(param);
    return success();
  }

  LogicalResult visit(const slang::ast::SpecparamSymbol &param) {
    visitParameter(param);
    return success();
  }

  template <class Node>
  void visitParameter(const Node &param) {
    // If debug info is enabled, try to materialize the parameter's constant
    // value on a best-effort basis and create a `dbg.variable` to track the
    // value.
    if (!context.options.debugInfo)
      return;
    auto value =
        context.materializeConstant(param.getValue(), param.getType(), loc);
    if (!value)
      return;
    if (builder.getInsertionBlock()->getParentOp() == context.intoModuleOp)
      context.orderedRootOps.insert({param.location, value.getDefiningOp()});

    // Prefix the parameter name with the surrounding namespace to create
    // somewhat sane names in the IR.
    SmallString<64> paramName;
    guessNamespacePrefix(param.getParentScope()->asSymbol(), paramName);
    paramName += param.name;

    debug::VariableOp::create(builder, loc, builder.getStringAttr(paramName),
                              value, Value{});
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Top-Level Item Conversion
//===----------------------------------------------------------------------===//

namespace {
struct RootVisitor : public BaseVisitor {
  using BaseVisitor::BaseVisitor;
  using BaseVisitor::visit;

  // Ignore standalone class declarations.
  LogicalResult visit(const slang::ast::ClassType &) { return success(); }
  LogicalResult visit(const slang::ast::GenericClassDefSymbol &) {
    return success();
  }

  // Handle packages.
  LogicalResult visit(const slang::ast::PackageSymbol &package) {
    return context.convertPackage(package);
  }

  // Handle functions and tasks.
  LogicalResult visit(const slang::ast::SubroutineSymbol &subroutine) {
    return context.convertFunction(subroutine);
  }

  // Handle top-level variables (e.g., global labels used in testbenches).
  LogicalResult visit(const slang::ast::VariableSymbol &varNode) {
    auto loweredType = context.convertType(*varNode.getDeclaredType());
    if (!loweredType)
      return failure();

    OpBuilder::InsertionGuard guard(builder);

    Value initial;
    if (const auto *init = varNode.getInitializer()) {
      initial = context.convertRvalueExpression(*init, loweredType);
      if (!initial)
        return failure();
    }

    auto it = context.orderedRootOps.upper_bound(varNode.location);
    if (it == context.orderedRootOps.end())
      builder.setInsertionPointToEnd(context.intoModuleOp.getBody());
    else
      builder.setInsertionPoint(it->second);

    auto varOp = moore::VariableOp::create(
        builder, loc,
        moore::RefType::get(cast<moore::UnpackedType>(loweredType)),
        builder.getStringAttr(varNode.name), initial);
    context.orderedRootOps.insert(it, {varNode.location, varOp});
    context.valueSymbols.insert(&varNode, varOp);
    return success();
  }

  // Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported construct: ")
        << slang::ast::toString(node.kind);
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Package Conversion
//===----------------------------------------------------------------------===//

namespace {
struct PackageVisitor : public BaseVisitor {
  using BaseVisitor::BaseVisitor;
  using BaseVisitor::visit;

  // Ignore class declarations inside packages. Full class lowering happens
  // elsewhere; packages often host stubs that do not require importer output.
  LogicalResult visit(const slang::ast::ClassType &) { return success(); }
  LogicalResult visit(const slang::ast::GenericClassDefSymbol &) {
    return success();
  }

  // Handle functions and tasks.
  LogicalResult visit(const slang::ast::SubroutineSymbol &subroutine) {
    if (const auto *parentScope = subroutine.getParentScope()) {
      const auto &parentSym = parentScope->asSymbol();
      if (parentSym.kind == slang::ast::SymbolKind::ClassType ||
          parentSym.kind == slang::ast::SymbolKind::GenericClassDef)
        return success();
    }
    return context.convertFunction(subroutine);
  }

  /// Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported package member: ")
        << slang::ast::toString(node.kind);
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Module Conversion
//===----------------------------------------------------------------------===//

static moore::ProcedureKind
convertProcedureKind(slang::ast::ProceduralBlockKind kind) {
  switch (kind) {
  case slang::ast::ProceduralBlockKind::Always:
    return moore::ProcedureKind::Always;
  case slang::ast::ProceduralBlockKind::AlwaysComb:
    return moore::ProcedureKind::AlwaysComb;
  case slang::ast::ProceduralBlockKind::AlwaysLatch:
    return moore::ProcedureKind::AlwaysLatch;
  case slang::ast::ProceduralBlockKind::AlwaysFF:
    return moore::ProcedureKind::AlwaysFF;
  case slang::ast::ProceduralBlockKind::Initial:
    return moore::ProcedureKind::Initial;
  case slang::ast::ProceduralBlockKind::Final:
    return moore::ProcedureKind::Final;
  }
  llvm_unreachable("all procedure kinds handled");
}

static moore::NetKind convertNetKind(slang::ast::NetType::NetKind kind) {
  switch (kind) {
  case slang::ast::NetType::Supply0:
    return moore::NetKind::Supply0;
  case slang::ast::NetType::Supply1:
    return moore::NetKind::Supply1;
  case slang::ast::NetType::Tri:
    return moore::NetKind::Tri;
  case slang::ast::NetType::TriAnd:
    return moore::NetKind::TriAnd;
  case slang::ast::NetType::TriOr:
    return moore::NetKind::TriOr;
  case slang::ast::NetType::TriReg:
    return moore::NetKind::TriReg;
  case slang::ast::NetType::Tri0:
    return moore::NetKind::Tri0;
  case slang::ast::NetType::Tri1:
    return moore::NetKind::Tri1;
  case slang::ast::NetType::UWire:
    return moore::NetKind::UWire;
  case slang::ast::NetType::Wire:
    return moore::NetKind::Wire;
  case slang::ast::NetType::WAnd:
    return moore::NetKind::WAnd;
  case slang::ast::NetType::WOr:
    return moore::NetKind::WOr;
  case slang::ast::NetType::Interconnect:
    return moore::NetKind::Interconnect;
  case slang::ast::NetType::UserDefined:
    return moore::NetKind::UserDefined;
  case slang::ast::NetType::Unknown:
    return moore::NetKind::Unknown;
  }
  llvm_unreachable("all net kinds handled");
}

namespace {
struct ModuleVisitor : public BaseVisitor {
  using BaseVisitor::visit;

  // A prefix of block names such as `foo.bar.` to put in front of variable and
  // instance names.
  StringRef blockNamePrefix;

  ModuleVisitor(Context &context, Location loc, StringRef blockNamePrefix = "")
      : BaseVisitor(context, loc), blockNamePrefix(blockNamePrefix) {}

  // Skip ports which are already handled by the module itself.
  LogicalResult visit(const slang::ast::PortSymbol &) { return success(); }
  LogicalResult visit(const slang::ast::MultiPortSymbol &) { return success(); }

  // Skip class declarations nested in modules.
  LogicalResult visit(const slang::ast::ClassType &) { return success(); }
  LogicalResult visit(const slang::ast::GenericClassDefSymbol &) {
    return success();
  }

  // Ignore interface port declarations that appear in module bodies (these
  // are handled as part of the module header lowering).
  LogicalResult visit(const slang::ast::InterfacePortSymbol &) {
    return success();
  }

  // Skip genvars.
  LogicalResult visit(const slang::ast::GenvarSymbol &genvarNode) {
    return success();
  }

  // Skip defparams which have been handled by slang.
  LogicalResult visit(const slang::ast::DefParamSymbol &) { return success(); }

  // Ignore type parameters. These have already been handled by Slang's type
  // checking.
  LogicalResult visit(const slang::ast::TypeParameterSymbol &) {
    return success();
  }

  // Handle instances.
  LogicalResult visit(const slang::ast::InstanceSymbol &instNode) {
    using slang::ast::ArgumentDirection;
    using slang::ast::AssignmentExpression;
    using slang::ast::MultiPortSymbol;
    using slang::ast::PortSymbol;

    switch (instNode.body.getDefinition().definitionKind) {
    case slang::ast::DefinitionKind::Interface: {
      auto *lowering = context.convertInterface(&instNode.body);
      if (!lowering)
        return failure();
      auto inst = builder.create<sv::InterfaceInstanceOp>(loc, lowering->type);
      inst.getOperation()->setAttr(
          "name", builder.getStringAttr(Twine(blockNamePrefix) + instNode.name));
      context.valueSymbols.insert(&instNode, inst.getResult());
      return success();
    }
    case slang::ast::DefinitionKind::Module:
      break;
    default:
      mlir::emitError(loc, "unsupported instance of ")
          << instNode.body.getDefinition().getKindString();
      return failure();
    }

    auto *moduleLowering = context.convertModuleHeader(&instNode.body);
    if (!moduleLowering)
      return failure();
    auto module = moduleLowering->op;
    auto moduleType = module.getModuleType();

    // Set visibility attribute for instantiated module.
    SymbolTable::setSymbolVisibility(module, SymbolTable::Visibility::Private);

    // Prepare the values that are involved in port connections. This creates
    // rvalues for input ports and appropriate lvalues for output, inout, and
    // ref ports. We also separate multi-ports into the individual underlying
    // ports with their corresponding connection.
    SmallDenseMap<const PortSymbol *, Value> portValues;
    portValues.reserve(moduleType.getNumPorts());
    SmallDenseMap<const slang::ast::InterfacePortSymbol *, Value>
        interfacePortValues;

    for (const auto *con : instNode.getPortConnections()) {
      const auto *expr = con->getExpression();

      // Handle unconnected behavior. The expression is null if it have no
      // connection for the port.
      if (!expr) {
        auto *port = con->port.as_if<PortSymbol>();
        if (auto *existingPort =
                moduleLowering->portsBySyntaxNode.lookup(port->getSyntax()))
          port = existingPort;

        switch (port->direction) {
        case ArgumentDirection::In: {
          auto refType = moore::RefType::get(
              cast<moore::UnpackedType>(context.convertType(port->getType())));

          if (const auto *net =
                  port->internalSymbol->as_if<slang::ast::NetSymbol>()) {
            auto netOp = moore::NetOp::create(
                builder, loc, refType,
                StringAttr::get(builder.getContext(), net->name),
                convertNetKind(net->netType.netKind), nullptr);
            auto readOp = moore::ReadOp::create(builder, loc, netOp);
            portValues.insert({port, readOp});
          } else if (const auto *var =
                         port->internalSymbol
                             ->as_if<slang::ast::VariableSymbol>()) {
            auto varOp = moore::VariableOp::create(
                builder, loc, refType,
                StringAttr::get(builder.getContext(), var->name), nullptr);
            auto readOp = moore::ReadOp::create(builder, loc, varOp);
            portValues.insert({port, readOp});
          } else {
            return mlir::emitError(loc)
                   << "unsupported internal symbol for unconnected port `"
                   << port->name << "`";
          }
          continue;
        }

        // No need to express unconnected behavior for output port, skip to the
        // next iteration of the loop.
        case ArgumentDirection::Out:
          continue;

        // TODO: Mark Inout port as unsupported and it will be supported later.
        default:
          return mlir::emitError(loc)
                 << "unsupported port `" << port->name << "` ("
                 << slang::ast::toString(port->kind) << ")";
        }
      }

      // Unpack the `<expr> = EmptyArgument` pattern emitted by Slang for
      // output and inout ports.
      if (const auto *assign = expr->as_if<AssignmentExpression>())
        expr = &assign->left();

      // Regular ports lower the connected expression to an lvalue or rvalue and
      // either attach it to the instance as an operand (for input, inout, and
      // ref ports), or assign an instance output to it (for output ports).
      if (auto *port = con->port.as_if<PortSymbol>()) {
        // Convert as rvalue for inputs, lvalue for all others.
        auto value = (port->direction == ArgumentDirection::In)
                         ? context.convertRvalueExpression(*expr)
                         : context.convertLvalueExpression(*expr);
        if (!value)
          return failure();
        if (auto *existingPort =
                moduleLowering->portsBySyntaxNode.lookup(con->port.getSyntax()))
          port = existingPort;
        portValues.insert({port, value});
        continue;
      }

      if (const auto *ifacePort =
              con->port.as_if<slang::ast::InterfacePortSymbol>()) {
        Value value;
        if (expr) {
          value = context.convertRvalueExpression(*expr);
        } else {
          auto [connectedSym, modportSym] = ifacePort->getConnection();
          if (connectedSym)
            value = context.valueSymbols.lookup(connectedSym);
          if (!value) {
            auto portLoc = context.convertLocation(ifacePort->location);
            return mlir::emitError(portLoc)
                   << "interface port `" << ifacePort->name
                   << "` lacks a connection expression";
          }
          if (modportSym) {
            auto modportAttr =
                context.lookupInterfaceModport(*modportSym, loc);
            if (failed(modportAttr))
              return failure();
            if (!isa<sv::ModportType>(value.getType())) {
              auto modportType =
                  sv::ModportType::get(builder.getContext(), *modportAttr);
              auto fieldAttr = FlatSymbolRefAttr::get(builder.getContext(),
                                                      modportSym->name);
              value = builder
                          .create<sv::GetModportOp>(loc, modportType, value,
                                                    fieldAttr)
                          .getResult();
            }
          }
        }
        if (!value)
          return failure();
        interfacePortValues.insert({ifacePort, value});
        continue;
      }

      // Multi-ports lower the connected expression to an lvalue and then slice
      // it up into multiple sub-values, one for each of the ports in the
      // multi-port.
      if (const auto *multiPort = con->port.as_if<MultiPortSymbol>()) {
        // Convert as lvalue.
        auto value = context.convertLvalueExpression(*expr);
        if (!value)
          return failure();
        unsigned offset = 0;
        for (const auto *port : llvm::reverse(multiPort->ports)) {
          if (auto *existingPort = moduleLowering->portsBySyntaxNode.lookup(
                  con->port.getSyntax()))
            port = existingPort;
          unsigned width = port->getType().getBitWidth();
          auto sliceType = context.convertType(port->getType());
          if (!sliceType)
            return failure();
          Value slice = moore::ExtractRefOp::create(
              builder, loc,
              moore::RefType::get(cast<moore::UnpackedType>(sliceType)), value,
              offset);
          // Create the "ReadOp" for input ports.
          if (port->direction == ArgumentDirection::In)
            slice = moore::ReadOp::create(builder, loc, slice);
          portValues.insert({port, slice});
          offset += width;
        }
        continue;
      }

      mlir::emitError(loc) << "unsupported instance port `" << con->port.name
                           << "` (" << slang::ast::toString(con->port.kind)
                           << ")";
      return failure();
    }

    // Match the module's ports up with the port values determined above.
    SmallVector<Value> inputValues;
    SmallVector<Value> outputValues;
    inputValues.reserve(moduleType.getNumInputs());
    outputValues.reserve(moduleType.getNumOutputs());

    for (auto &info : moduleLowering->portInfos) {
      if (info.kind == ModuleLowering::ModulePortInfo::Kind::Data) {
        auto &port = moduleLowering->ports[info.index];
        auto value = portValues.lookup(&port.ast);
        if (port.ast.direction == ArgumentDirection::Out) {
          outputValues.push_back(value);
        } else {
          if (!value)
            return mlir::emitError(loc) << "missing connection for port `"
                                        << port.ast.name << "`";
          inputValues.push_back(value);
        }
        continue;
      }
      auto &ifacePort = moduleLowering->interfacePorts[info.index];
      auto value = interfacePortValues.lookup(&ifacePort.ast);
      if (!value)
        value = context.valueSymbols.lookup(&ifacePort.ast);
      if (!value)
        return mlir::emitError(loc) << "missing connection for interface port `"
                                    << ifacePort.ast.name << "`";
      inputValues.push_back(value);
    }

    auto inputTypes = moduleType.getInputTypes();
    auto inputNameVec = moduleType.getInputNames();
    auto outputNameVec = moduleType.getOutputNames();
    for (auto [idx, type] : llvm::enumerate(inputTypes)) {
      auto &value = inputValues[idx];
      if (!value) {
        auto nameAttr = llvm::dyn_cast<StringAttr>(inputNameVec[idx]);
        auto name = nameAttr ? nameAttr.getValue() : StringRef("<unnamed>");
        return mlir::emitError(loc)
               << "missing value for input port `" << name << "`";
      }
      if (isa<sv::InterfaceType>(type) || isa<sv::ModportType>(type))
        continue;
      // TODO: This should honor signedness in the conversion.
      value =
          context.materializeConversion(type, value, false, value.getLoc());
    }

    // Here we use the hierarchical value recorded in `Context::valueSymbols`.
    // Then we pass it as the input port with the ref<T> type of the instance.
    for (const auto &hierPath : context.hierPaths[&instNode.body])
      if (auto hierValue = context.valueSymbols.lookup(hierPath.valueSym);
          hierPath.hierName && hierPath.direction == ArgumentDirection::In)
        inputValues.push_back(hierValue);

    // Create the instance op itself.
    SmallVector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr(
        "instanceName",
        builder.getStringAttr(Twine(blockNamePrefix) + instNode.name)));
    attrs.push_back(builder.getNamedAttr(
        "moduleName", FlatSymbolRefAttr::get(module.getSymNameAttr())));
    attrs.push_back(builder.getNamedAttr("inputNames",
                                         builder.getArrayAttr(inputNameVec)));
    attrs.push_back(builder.getNamedAttr(
        "outputNames", builder.getArrayAttr(outputNameVec)));
    auto inst = builder.create<moore::InstanceOp>(
        loc, moduleType.getOutputTypes(), inputValues, attrs);

    // Record instance's results generated by hierarchical names.
    for (const auto &hierPath : context.hierPaths[&instNode.body])
      if (hierPath.idx && hierPath.direction == ArgumentDirection::Out)
        context.valueSymbols.insert(hierPath.valueSym,
                                    inst->getResult(*hierPath.idx));

    // Assign output values from the instance to the connected expression.
    for (auto [lvalue, output] : llvm::zip(outputValues, inst.getOutputs())) {
      if (!lvalue)
        continue;
      Value rvalue = output;
      auto dstType = cast<moore::RefType>(lvalue.getType()).getNestedType();
      // TODO: This should honor signedness in the conversion.
      rvalue = context.materializeConversion(dstType, rvalue, false, loc);
      moore::ContinuousAssignOp::create(builder, loc, lvalue, rvalue);
    }

    return success();
  }

  // Handle variables.
  LogicalResult visit(const slang::ast::VariableSymbol &varNode) {
    auto loweredType = context.convertType(*varNode.getDeclaredType());
    if (!loweredType)
      return failure();

    Value initial;
    if (const auto *init = varNode.getInitializer()) {
      initial = context.convertRvalueExpression(*init, loweredType);
      if (!initial)
        return failure();
    }

    auto varOp = moore::VariableOp::create(
        builder, loc,
        moore::RefType::get(cast<moore::UnpackedType>(loweredType)),
        builder.getStringAttr(Twine(blockNamePrefix) + varNode.name), initial);
    context.valueSymbols.insert(&varNode, varOp);
    return success();
  }

  // Handle nets.
  LogicalResult visit(const slang::ast::NetSymbol &netNode) {
    auto loweredType = context.convertType(*netNode.getDeclaredType());
    if (!loweredType)
      return failure();

    Value assignment;
    if (const auto *init = netNode.getInitializer()) {
      assignment = context.convertRvalueExpression(*init, loweredType);
      if (!assignment)
        return failure();
    }

    auto netkind = convertNetKind(netNode.netType.netKind);
    if (netkind == moore::NetKind::Interconnect ||
        netkind == moore::NetKind::UserDefined ||
        netkind == moore::NetKind::Unknown)
      return mlir::emitError(loc, "unsupported net kind `")
             << netNode.netType.name << "`";

    auto netOp = moore::NetOp::create(
        builder, loc,
        moore::RefType::get(cast<moore::UnpackedType>(loweredType)),
        builder.getStringAttr(Twine(blockNamePrefix) + netNode.name), netkind,
        assignment);
    context.valueSymbols.insert(&netNode, netOp);
    return success();
  }

  // Handle continuous assignments.
  LogicalResult visit(const slang::ast::ContinuousAssignSymbol &assignNode) {
    if (const auto *delay = assignNode.getDelay()) {
      auto loc = context.convertLocation(delay->sourceRange);
      return mlir::emitError(loc,
                             "delayed continuous assignments not supported");
    }

    const auto &expr =
        assignNode.getAssignment().as<slang::ast::AssignmentExpression>();
    if (const auto *member =
            expr.left().as_if<slang::ast::MemberAccessExpression>()) {
      auto ifaceValue = context.convertRvalueExpression(member->value());
      if (ifaceValue && isa<sv::InterfaceType>(ifaceValue.getType())) {
        auto assigned =
            context.assignInterfaceMember(expr.left(), expr.right(), loc);
        if (failed(assigned))
          return failure();
        return success();
      }
    } else if (const auto *hier =
                   expr.left().as_if<slang::ast::HierarchicalValueExpression>()) {
      if (hier->ref.isViaIfacePort()) {
        auto assigned =
            context.assignInterfaceMember(expr.left(), expr.right(), loc);
        if (failed(assigned))
          return failure();
        return success();
      }
    }

    auto lhs = context.convertLvalueExpression(expr.left());
    if (!lhs)
      return failure();

    auto rhs = context.convertRvalueExpression(
        expr.right(), cast<moore::RefType>(lhs.getType()).getNestedType());
    if (!rhs)
      return failure();

    moore::ContinuousAssignOp::create(builder, loc, lhs, rhs);
    return success();
  }

  // Handle procedures.
  LogicalResult convertProcedure(moore::ProcedureKind kind,
                                 const slang::ast::Statement &body) {
    auto procOp = moore::ProcedureOp::create(builder, loc, kind);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&procOp.getBody().emplaceBlock());
    Context::ValueSymbolScope scope(context.valueSymbols);
    if (failed(context.convertStatement(body)))
      return failure();
    if (builder.getBlock())
      moore::ReturnOp::create(builder, loc);
    return success();
  }

  LogicalResult visit(const slang::ast::ProceduralBlockSymbol &procNode) {
    // Detect `always @(*) <stmt>` and convert to `always_comb <stmt>` if
    // requested by the user.
    if (context.options.lowerAlwaysAtStarAsComb) {
      auto *stmt = procNode.getBody().as_if<slang::ast::TimedStatement>();
      if (procNode.procedureKind == slang::ast::ProceduralBlockKind::Always &&
          stmt &&
          stmt->timing.kind == slang::ast::TimingControlKind::ImplicitEvent)
        return convertProcedure(moore::ProcedureKind::AlwaysComb, stmt->stmt);
    }

    return convertProcedure(convertProcedureKind(procNode.procedureKind),
                            procNode.getBody());
  }

  // Handle generate block.
  LogicalResult visit(const slang::ast::GenerateBlockSymbol &genNode) {
    // Ignore uninstantiated blocks.
    if (genNode.isUninstantiated)
      return success();

    // If the block has a name, add it to the list of block name prefices.
    SmallString<64> prefixBuffer;
    auto prefix = blockNamePrefix;
    if (!genNode.name.empty()) {
      prefixBuffer += blockNamePrefix;
      prefixBuffer += genNode.name;
      prefixBuffer += '.';
      prefix = prefixBuffer;
    }

    // Visit each member of the generate block.
    for (auto &member : genNode.members())
      if (failed(member.visit(ModuleVisitor(context, loc, prefix))))
        return failure();
    return success();
  }

  // Handle generate block array.
  LogicalResult visit(const slang::ast::GenerateBlockArraySymbol &genArrNode) {
    // If the block has a name, add it to the list of block name prefices and
    // prepare to append the array index and a `.` in each iteration.
    SmallString<64> prefixBuffer;
    if (!genArrNode.name.empty()) {
      prefixBuffer += blockNamePrefix;
      prefixBuffer += genArrNode.name;
    }
    auto prefixBufferBaseLen = prefixBuffer.size();

    // Visit each iteration entry of the generate block.
    for (const auto *entry : genArrNode.entries) {
      // Append the index to the prefix if this block has a name.
      auto prefix = blockNamePrefix;
      if (prefixBufferBaseLen > 0) {
        prefixBuffer.resize(prefixBufferBaseLen);
        prefixBuffer += '_';
        if (entry->arrayIndex)
          prefixBuffer += entry->arrayIndex->toString();
        else
          Twine(entry->constructIndex).toVector(prefixBuffer);
        prefixBuffer += '.';
        prefix = prefixBuffer;
      }

      // Visit this iteration entry.
      if (failed(entry->asSymbol().visit(ModuleVisitor(context, loc, prefix))))
        return failure();
    }
    return success();
  }

  // Ignore statement block symbols. These get generated by Slang for blocks
  // with variables and other declarations. For example, having an initial
  // procedure with a variable declaration, such as `initial begin int x;
  // end`, will create the procedure with a block and variable declaration as
  // expected, but will also create a `StatementBlockSymbol` with just the
  // variable layout _next to_ the initial procedure.
  LogicalResult visit(const slang::ast::StatementBlockSymbol &) {
    return success();
  }

  // Ignore sequence declarations. The declarations are already evaluated by
  // Slang and are part of an AssertionInstance.
  LogicalResult visit(const slang::ast::SequenceSymbol &seqNode) {
    return success();
  }

  // Ignore property declarations. The declarations are already evaluated by
  // Slang and are part of an AssertionInstance.
  LogicalResult visit(const slang::ast::PropertySymbol &propNode) {
    return success();
  }

  // Handle functions and tasks.
  LogicalResult visit(const slang::ast::SubroutineSymbol &subroutine) {
    return context.convertFunction(subroutine);
  }

  /// Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported module member: ")
        << slang::ast::toString(node.kind);
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Structure and Hierarchy Conversion
//===----------------------------------------------------------------------===//

/// Convert an entire Slang compilation to MLIR ops. This is the main entry
/// point for the conversion.
LogicalResult Context::convertCompilation() {
  const auto &root = compilation.getRoot();

  // Keep track of the local time scale. `getTimeScale` automatically looks
  // through parent scopes to find the time scale effective locally.
  auto prevTimeScale = timeScale;
  timeScale = root.getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard =
      llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  // Maintain a root scope for value symbols so top-level declarations remain
  // visible to downstream modules and packages.
  ValueSymbolScope rootScope(valueSymbols);

  // First only to visit the whole AST to collect the hierarchical names without
  // any operation creating.
  for (auto *inst : root.topInstances)
    if (failed(traverseInstanceBody(inst->body)))
      return failure();

  // Visit all top-level declarations in all compilation units. This does not
  // include instantiable constructs like modules, interfaces, and programs,
  // which are listed separately as top instances.
  for (auto *unit : root.compilationUnits) {
    for (const auto &member : unit->members()) {
      auto loc = convertLocation(member.location);
      if (failed(member.visit(RootVisitor(*this, loc))))
        return failure();
    }
  }

  // Prime the root definition worklist by adding all the top-level modules and
  // materialize interface definitions that may be referenced later.
  for (auto *inst : root.topInstances) {
    auto *body = &inst->body;
    switch (body->getDefinition().definitionKind) {
    case slang::ast::DefinitionKind::Module:
      if (!convertModuleHeader(body))
        return failure();
      break;
    case slang::ast::DefinitionKind::Interface:
      if (!convertInterface(body))
        return failure();
      break;
    default:
      break;
    }
  }

  // Convert all the root module definitions.
  while (!moduleWorklist.empty()) {
    auto *module = moduleWorklist.front();
    moduleWorklist.pop();
    if (failed(convertModuleBody(module)))
      return failure();
  }

  return success();
}

InterfaceLowering *
Context::convertInterface(const slang::ast::InstanceBodySymbol *interface) {
  using slang::ast::ArgumentDirection;

  auto &slot = interfaces[interface];
  if (slot)
    return slot.get();

  if (interface->getDefinition().definitionKind !=
      slang::ast::DefinitionKind::Interface) {
    auto loc = convertLocation(interface->location);
    mlir::emitError(loc) << "expected interface definition but got "
                         << interface->getDefinition().getKindString();
    return {};
  }

  slot = std::make_unique<InterfaceLowering>();
  auto &lowering = *slot;

  auto loc = convertLocation(interface->location);

  OpBuilder::InsertionGuard guard(builder);
  auto it = orderedRootOps.upper_bound(interface->location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  auto ifaceOp = sv::InterfaceOp::create(builder, loc, interface->name);
  orderedRootOps.insert(it, {interface->location, ifaceOp});
  symbolTable.insert(ifaceOp);
  lowering.op = ifaceOp;
  lowering.type = sv::InterfaceType::get(
      builder.getContext(),
      FlatSymbolRefAttr::get(builder.getContext(), ifaceOp.getSymNameAttr()));

  auto *bodyBlock = ifaceOp.getBodyBlock();
  OpBuilder ifaceBuilder(bodyBlock, bodyBlock->begin());

  auto emitSignal = [&](const slang::ast::ValueSymbol &value) -> LogicalResult {
    if (lowering.signalRefs.contains(&value))
      return success();
    auto type = convertType(*value.getDeclaredType());
    if (!type)
      return failure();
    auto sigLoc = convertLocation(value.location);
    Type signalType;
    if (auto intType = dyn_cast<moore::IntType>(type)) {
      unsigned width = intType.getBitSize().value_or(1);
      if (width == 0)
        width = 1;
      auto widthAttr = IntegerAttr::get(
          IntegerType::get(ifaceBuilder.getContext(), 32), APInt(32, width));
      signalType = hw::IntType::get(widthAttr);
    } else {
      mlir::emitError(sigLoc) << "unsupported interface signal type " << type;
      return failure();
    }
    auto signalOp = ifaceBuilder.create<sv::InterfaceSignalOp>(
        sigLoc, ifaceBuilder.getStringAttr(value.name), signalType);
    auto symRef = FlatSymbolRefAttr::get(ifaceBuilder.getContext(),
                                         signalOp.getSymNameAttr());
    lowering.signalRefs.try_emplace(&value, symRef);
    lowering.signalRefsByName.try_emplace(signalOp.getSymNameAttr(), symRef);
    return success();
  };

  auto convertDirection = [&](ArgumentDirection direction)
      -> std::optional<sv::ModportDirection> {
    switch (direction) {
    case ArgumentDirection::In:
      return sv::ModportDirection::input;
    case ArgumentDirection::Out:
      return sv::ModportDirection::output;
    case ArgumentDirection::InOut:
      return sv::ModportDirection::inout;
    default:
      return std::nullopt;
    }
  };

  for (auto &member : interface->members()) {
    if (auto *var = member.as_if<slang::ast::VariableSymbol>()) {
      if (failed(emitSignal(*var)))
        return {};
      continue;
    }
    if (auto *net = member.as_if<slang::ast::NetSymbol>()) {
      if (failed(emitSignal(*net)))
        return {};
      continue;
    }
    if (auto *port = member.as_if<slang::ast::PortSymbol>()) {
      if (auto *internal =
              port->internalSymbol
                  ? port->internalSymbol->as_if<slang::ast::ValueSymbol>()
                  : nullptr)
        if (failed(emitSignal(*internal)))
          return {};
      continue;
    }
    if (auto *modport = member.as_if<slang::ast::ModportSymbol>()) {
      SmallVector<Attribute> portEntries;
      for (auto &mpMember :
           modport->membersOfType<slang::ast::ModportPortSymbol>()) {
        auto direction = convertDirection(mpMember.direction);
        if (!direction) {
          auto mpLoc = convertLocation(mpMember.location);
          mlir::emitError(mpLoc)
              << "unsupported modport direction "
              << slang::ast::toString(mpMember.direction);
          return {};
        }
        const slang::ast::ValueSymbol *target = nullptr;
        if (mpMember.internalSymbol)
          target = mpMember.internalSymbol->as_if<slang::ast::ValueSymbol>();
        if (!target && mpMember.internalSymbol)
          if (const auto *portSym =
                  mpMember.internalSymbol->as_if<slang::ast::PortSymbol>())
            if (portSym->internalSymbol)
              target =
                  portSym->internalSymbol->as_if<slang::ast::ValueSymbol>();
        FlatSymbolRefAttr symRef;
        if (!target) {
          auto mpLoc = convertLocation(mpMember.location);
          auto nameAttr = ifaceBuilder.getStringAttr(mpMember.name);
          symRef = lowering.signalRefsByName.lookup(nameAttr);
          if (!symRef) {
            mlir::emitError(mpLoc)
                << "modport member `" << mpMember.name
                << "` does not reference a value symbol";
            return {};
          }
        } else {
          if (failed(emitSignal(*target)))
            return {};
          symRef = lowering.signalRefs.lookup(target);
          if (!symRef) {
            auto mpLoc = convertLocation(mpMember.location);
            mlir::emitError(mpLoc)
                << "internal interface signal `" << target->name
                << "` was not materialized";
            return {};
          }
        }
        auto dirAttr = sv::ModportDirectionAttr::get(
            ifaceBuilder.getContext(), *direction);
        portEntries.push_back(
            sv::ModportStructAttr::get(ifaceBuilder.getContext(), dirAttr,
                                       symRef));
      }
      auto portsAttr = ifaceBuilder.getArrayAttr(portEntries);
      auto modportOp = ifaceBuilder.create<sv::InterfaceModportOp>(
          convertLocation(modport->location),
          ifaceBuilder.getStringAttr(modport->name), portsAttr);
      lowering.modportRefs.try_emplace(
          modport,
          SymbolRefAttr::get(
              ifaceBuilder.getContext(), ifaceOp.getSymNameAttr(),
              FlatSymbolRefAttr::get(ifaceBuilder.getContext(),
                                     modportOp.getSymNameAttr())));
      continue;
    }
  }

  return &lowering;
}

FailureOr<InterfaceLowering *>
Context::ensureInterfaceLowering(const slang::ast::Scope &scope,
                                 Location loc) {
  if (auto *instanceBody = scope.getContainingInstance())
    if (auto *lowering = convertInterface(instanceBody))
      return lowering;

  auto &scopeSymbol = scope.asSymbol();
  if (scopeSymbol.kind == slang::ast::SymbolKind::InterfacePort) {
    if (auto *parentScope = scopeSymbol.getParentScope())
      if (auto *instanceBody = parentScope->getContainingInstance())
        if (auto *lowering = convertInterface(instanceBody))
          return lowering;
  }

  mlir::emitError(loc)
      << "expected interface instance scope but found "
      << slang::ast::toString(scopeSymbol.kind);
  return failure();
}

FailureOr<FlatSymbolRefAttr>
Context::lookupInterfaceSignal(const slang::ast::ValueSymbol &symbol,
                               Location loc) {
  auto parentScope = symbol.getParentScope();
  if (!parentScope)
    return mlir::emitError(loc)
           << "interface member `" << symbol.name
           << "` does not have a parent scope";
  auto lowering = ensureInterfaceLowering(*parentScope, loc);
  if (failed(lowering))
    return failure();
  if (auto attr = (*lowering)->signalRefs.lookup(&symbol))
    return attr;
  auto nameAttr = builder.getStringAttr(symbol.name);
  if (auto attr = (*lowering)->signalRefsByName.lookup(nameAttr))
    return attr;
  return mlir::emitError(loc)
         << "missing lowered signal for interface member `" << symbol.name
         << "`";
}

FailureOr<mlir::SymbolRefAttr>
Context::lookupInterfaceModport(const slang::ast::ModportSymbol &symbol,
                                Location loc) {
  auto parentScope = symbol.getParentScope();
  if (!parentScope)
    return mlir::emitError(loc)
           << "modport `" << symbol.name << "` does not have a parent scope";
  auto lowering = ensureInterfaceLowering(*parentScope, loc);
  if (failed(lowering))
    return failure();
  if (auto attr = (*lowering)->modportRefs.lookup(&symbol))
    return attr;
  return mlir::emitError(loc)
         << "missing lowered modport for `" << symbol.name << "`";
}

/// Convert a module and its ports to an empty module op in the IR. Also adds
/// the op to the worklist of module bodies to be lowered. This acts like a
/// module "declaration", allowing instances to already refer to a module even
/// before its body has been lowered.
ModuleLowering *
Context::convertModuleHeader(const slang::ast::InstanceBodySymbol *module) {
  using slang::ast::ArgumentDirection;
  using slang::ast::MultiPortSymbol;
  using slang::ast::ParameterSymbol;
  using slang::ast::PortSymbol;
  using slang::ast::TypeParameterSymbol;

  // Keep track of the local time scale. `getTimeScale` automatically looks
  // through parent scopes to find the time scale effective locally.
  auto prevTimeScale = timeScale;
  timeScale = module->getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard =
      llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  auto parameters = module->getParameters();
  bool hasModuleSame = false;
  // If there is already exist a module that has the same name with this
  // module ,has the same parent scope and has the same parameters we can
  // define this module is a duplicate module
  for (auto const &existingModule : modules) {
    if (module->getDeclaringDefinition() ==
        existingModule.getFirst()->getDeclaringDefinition()) {
      auto moduleParameters = existingModule.getFirst()->getParameters();
      hasModuleSame = true;
      for (auto it1 = parameters.begin(), it2 = moduleParameters.begin();
           it1 != parameters.end() && it2 != moduleParameters.end();
           it1++, it2++) {
        // Parameters size different
        if (it1 == parameters.end() || it2 == moduleParameters.end()) {
          hasModuleSame = false;
          break;
        }
        const auto *para1 = (*it1)->symbol.as_if<ParameterSymbol>();
        const auto *para2 = (*it2)->symbol.as_if<ParameterSymbol>();
        // Parameters kind different
        if ((para1 == nullptr) ^ (para2 == nullptr)) {
          hasModuleSame = false;
          break;
        }
        // Compare ParameterSymbol
        if (para1 != nullptr) {
          hasModuleSame = para1->getValue() == para2->getValue();
        }
        // Compare TypeParameterSymbol
        if (para1 == nullptr) {
          auto para1Type = convertType(
              (*it1)->symbol.as<TypeParameterSymbol>().getTypeAlias());
          auto para2Type = convertType(
              (*it2)->symbol.as<TypeParameterSymbol>().getTypeAlias());
          hasModuleSame = para1Type == para2Type;
        }
        if (!hasModuleSame)
          break;
      }
      if (hasModuleSame) {
        module = existingModule.first;
        break;
      }
    }
  }

  auto &slot = modules[module];
  if (slot)
    return slot.get();
  slot = std::make_unique<ModuleLowering>();
  auto &lowering = *slot;

  auto loc = convertLocation(module->location);
  OpBuilder::InsertionGuard g(builder);

  // We only support modules for now. Extension to interfaces and programs
  // should be trivial though, since they are essentially the same thing with
  // only minor differences in semantics.
  if (module->getDefinition().definitionKind !=
      slang::ast::DefinitionKind::Module) {
    mlir::emitError(loc) << "unsupported definition: "
                         << module->getDefinition().getKindString();
    return {};
  }

  // Handle the port list.
  auto block = std::make_unique<Block>();
  SmallVector<hw::ModulePort> modulePorts;

  // It's used to tag where a hierarchical name is on the port list.
  unsigned int outputIdx = 0, inputIdx = 0;
  for (auto *symbol : module->getPortList()) {
    if (const auto *ifacePort =
            symbol->as_if<slang::ast::InterfacePortSymbol>()) {
      if (!ifacePort->interfaceDef || ifacePort->isGeneric) {
        auto portLoc = convertLocation(ifacePort->location);
        mlir::emitError(portLoc)
            << "unsupported interface port `" << ifacePort->name
            << "` without concrete interface definition";
        return {};
      }
      auto portLoc = convertLocation(ifacePort->location);
      auto ifaceRef = FlatSymbolRefAttr::get(builder.getContext(),
                                             ifacePort->interfaceDef->name);
      auto ifaceType = sv::InterfaceType::get(builder.getContext(), ifaceRef);
      auto portName = builder.getStringAttr(ifacePort->name);
      modulePorts.push_back(
          {portName, ifaceType, hw::ModulePort::Input});
      auto arg = block->addArgument(ifaceType, portLoc);
      lowering.interfacePorts.push_back({*ifacePort, portLoc, arg});
      lowering.portInfos.push_back(
          {ModuleLowering::ModulePortInfo::Kind::Interface,
           static_cast<unsigned>(lowering.interfacePorts.size() - 1)});
      inputIdx++;
      continue;
    }

    auto handlePort = [&](const PortSymbol &port) {
      auto portLoc = convertLocation(port.location);
      auto type = convertType(port.getType());
      if (!type)
        return failure();
      auto portName = builder.getStringAttr(port.name);
      BlockArgument arg;
      if (port.direction == ArgumentDirection::Out) {
        modulePorts.push_back({portName, type, hw::ModulePort::Output});
        outputIdx++;
      } else {
        // Only the ref type wrapper exists for the time being, the net type
        // wrapper for inout may be introduced later if necessary.
        if (port.direction != ArgumentDirection::In)
          type = moore::RefType::get(cast<moore::UnpackedType>(type));
        modulePorts.push_back({portName, type, hw::ModulePort::Input});
        arg = block->addArgument(type, portLoc);
        inputIdx++;
      }
      lowering.ports.push_back({port, portLoc, arg});
      lowering.portInfos.push_back(
          {ModuleLowering::ModulePortInfo::Kind::Data,
           static_cast<unsigned>(lowering.ports.size() - 1)});
      return success();
    };

    if (const auto *port = symbol->as_if<PortSymbol>()) {
      if (failed(handlePort(*port)))
        return {};
    } else if (const auto *multiPort = symbol->as_if<MultiPortSymbol>()) {
      for (auto *port : multiPort->ports)
        if (failed(handlePort(*port)))
          return {};
    } else {
      mlir::emitError(convertLocation(symbol->location))
          << "unsupported module port `" << symbol->name << "` ("
          << slang::ast::toString(symbol->kind) << ")";
      return {};
    }
  }

  // Mapping hierarchical names into the module's ports.
  for (auto &hierPath : hierPaths[module]) {
    auto hierType = convertType(hierPath.valueSym->getType());
    if (!hierType)
      return {};

    if (auto hierName = hierPath.hierName) {
      // The type of all hierarchical names are marked as the "RefType".
      hierType = moore::RefType::get(cast<moore::UnpackedType>(hierType));
      if (hierPath.direction == ArgumentDirection::Out) {
        hierPath.idx = outputIdx++;
        modulePorts.push_back({hierName, hierType, hw::ModulePort::Output});
      } else {
        hierPath.idx = inputIdx++;
        modulePorts.push_back({hierName, hierType, hw::ModulePort::Input});
        auto hierLoc = convertLocation(hierPath.valueSym->location);
        block->addArgument(hierType, hierLoc);
      }
    }
  }
  auto moduleType = hw::ModuleType::get(getContext(), modulePorts);

  // Pick an insertion point for this module according to the source file
  // location.
  auto it = orderedRootOps.upper_bound(module->location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Create an empty module that corresponds to this module.
  auto moduleOp =
      moore::SVModuleOp::create(builder, loc, module->name, moduleType);
  orderedRootOps.insert(it, {module->location, moduleOp});
  moduleOp.getBodyRegion().push_back(block.release());
  lowering.op = moduleOp;

  // Add the module to the symbol table of the MLIR module, which uniquifies its
  // name as we'd expect.
  symbolTable.insert(moduleOp);

  // Schedule the body to be lowered.
  moduleWorklist.push(module);

  // Map duplicate port by Syntax
  for (const auto &port : lowering.ports)
    lowering.portsBySyntaxNode.insert({port.ast.getSyntax(), &port.ast});

  return &lowering;
}

/// Convert a module's body to the corresponding IR ops. The module op must have
/// already been created earlier through a `convertModuleHeader` call.
LogicalResult
Context::convertModuleBody(const slang::ast::InstanceBodySymbol *module) {
  auto &lowering = *modules[module];
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(lowering.op.getBody());

  ValueSymbolScope scope(valueSymbols);
  for (auto &ifacePort : lowering.interfacePorts)
    valueSymbols.insert(&ifacePort.ast, ifacePort.arg);

  // Keep track of the local time scale. `getTimeScale` automatically looks
  // through parent scopes to find the time scale effective locally.
  auto prevTimeScale = timeScale;
  timeScale = module->getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard =
      llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  // Collect downward hierarchical names. Such as,
  // module SubA; int x = Top.y; endmodule. The "Top" module is the parent of
  // the "SubA", so "Top.y" is the downward hierarchical name.
  for (auto &hierPath : hierPaths[module])
    if (hierPath.direction == slang::ast::ArgumentDirection::In && hierPath.idx)
      valueSymbols.insert(hierPath.valueSym,
                          lowering.op.getBody()->getArgument(*hierPath.idx));

  // Convert the body of the module.
  for (auto &member : module->members()) {
    auto loc = convertLocation(member.location);
    if (failed(member.visit(ModuleVisitor(*this, loc))))
      return failure();
  }

  // Create additional ops to drive input port values onto the corresponding
  // internal variables and nets, and to collect output port values for the
  // terminator.
  SmallVector<Value> outputs;
  for (auto &port : lowering.ports) {
    Value value;
    if (auto *expr = port.ast.getInternalExpr()) {
      value = convertLvalueExpression(*expr);
    } else if (port.ast.internalSymbol) {
      if (const auto *sym =
              port.ast.internalSymbol->as_if<slang::ast::ValueSymbol>())
        value = valueSymbols.lookup(sym);
    }
    if (!value)
      return mlir::emitError(port.loc, "unsupported port: `")
             << port.ast.name
             << "` does not map to an internal symbol or expression";

    // Collect output port values to be returned in the terminator.
    if (port.ast.direction == slang::ast::ArgumentDirection::Out) {
      if (isa<moore::RefType>(value.getType()))
        value = moore::ReadOp::create(builder, value.getLoc(), value);
      outputs.push_back(value);
      continue;
    }

    // Assign the value coming in through the port to the internal net or symbol
    // of that port.
    Value portArg = port.arg;
    if (port.ast.direction != slang::ast::ArgumentDirection::In)
      portArg = moore::ReadOp::create(builder, port.loc, port.arg);
    moore::ContinuousAssignOp::create(builder, port.loc, value, portArg);
  }

  // Ensure the number of operands of this module's terminator and the number of
  // its(the current module) output ports remain consistent.
  for (auto &hierPath : hierPaths[module])
    if (auto hierValue = valueSymbols.lookup(hierPath.valueSym))
      if (hierPath.direction == slang::ast::ArgumentDirection::Out)
        outputs.push_back(hierValue);

  moore::OutputOp::create(builder, lowering.op.getLoc(), outputs);
  return success();
}

/// Convert a package and its contents.
LogicalResult
Context::convertPackage(const slang::ast::PackageSymbol &package) {
  // Keep track of the local time scale. `getTimeScale` automatically looks
  // through parent scopes to find the time scale effective locally.
  auto prevTimeScale = timeScale;
  timeScale = package.getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard =
      llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  // The lightweight UVM stub package only exists to satisfy preprocessing
  // requirements. Skip converting its contents so that unsupported class
  // constructs (and their methods) do not trigger importer crashes.
  if (package.name == "uvm_pkg")
    return success();

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(intoModuleOp.getBody());
  ValueSymbolScope scope(valueSymbols);
  for (auto &member : package.members()) {
    auto loc = convertLocation(member.location);
    if (failed(member.visit(PackageVisitor(*this, loc))))
      return failure();
  }
  return success();
}

/// Convert a function and its arguments to a function declaration in the IR.
/// This does not convert the function body.
FunctionLowering *
Context::declareFunction(const slang::ast::SubroutineSymbol &subroutine) {
  using slang::ast::ArgumentDirection;

  // Check if there already is a declaration for this function.
  auto &lowering = functions[&subroutine];
  if (lowering) {
    if (!lowering->op)
      return {};
    return lowering.get();
  }
  lowering = std::make_unique<FunctionLowering>();
  auto loc = convertLocation(subroutine.location);

  // Pick an insertion point for this function according to the source file
  // location.
  OpBuilder::InsertionGuard g(builder);
  auto it = orderedRootOps.upper_bound(subroutine.location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Class methods (including static ones) are currently lowered via
  // lightweight stubs instead of full-fledged functions. Record the
  // declaration as absent so callers can short-circuit appropriately.
  if (const auto *parentScope = subroutine.getParentScope()) {
    const auto &parentSym = parentScope->asSymbol();
    if (parentSym.kind == slang::ast::SymbolKind::ClassType) {
      lowering->op = nullptr;
      return lowering.get();
    }
    if (parentSym.kind == slang::ast::SymbolKind::Package &&
        parentSym.name == "uvm_pkg") {
      lowering->op = nullptr;
      return lowering.get();
    }
  } else if (subroutine.thisVar) {
    lowering->op = nullptr;
    return lowering.get();
  }

  // Determine the function type.
  SmallVector<Type> inputTypes;
  SmallVector<Type, 1> outputTypes;

  for (const auto *arg : subroutine.getArguments()) {
    auto type = convertType(arg->getType());
    if (!type)
      return {};
    if (arg->direction == ArgumentDirection::In) {
      inputTypes.push_back(type);
    } else {
      inputTypes.push_back(
          moore::RefType::get(cast<moore::UnpackedType>(type)));
    }
  }

  if (!subroutine.getReturnType().isVoid()) {
    auto type = convertType(subroutine.getReturnType());
    if (!type)
      return {};
    outputTypes.push_back(type);
  }

  auto funcType = FunctionType::get(getContext(), inputTypes, outputTypes);

  // Prefix the function name with the surrounding namespace to create somewhat
  // sane names in the IR.
  SmallString<64> funcName;
  guessNamespacePrefix(subroutine.getParentScope()->asSymbol(), funcName);
  funcName += subroutine.name;

  // Create a function declaration.
  auto funcOp = mlir::func::FuncOp::create(builder, loc, funcName, funcType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  orderedRootOps.insert(it, {subroutine.location, funcOp});
  lowering->op = funcOp;

  // Add the function to the symbol table of the MLIR module, which uniquifies
  // its name.
  symbolTable.insert(funcOp);

  return lowering.get();
}

/// Convert a function.
LogicalResult
Context::convertFunction(const slang::ast::SubroutineSymbol &subroutine) {
  if (subroutine.thisVar)
    return success();

  // Keep track of the local time scale. `getTimeScale` automatically looks
  // through parent scopes to find the time scale effective locally.
  auto prevTimeScale = timeScale;
  timeScale = subroutine.getTimeScale().value_or(slang::TimeScale());
  auto timeScaleGuard =
      llvm::make_scope_exit([&] { timeScale = prevTimeScale; });

  // First get or create the function declaration.
  auto *lowering = declareFunction(subroutine);
  if (!lowering)
    return failure();
  ValueSymbolScope scope(valueSymbols);

  // Create a function body block and populate it with block arguments.
  SmallVector<moore::VariableOp> argVariables;
  auto &block = lowering->op.getBody().emplaceBlock();
  for (auto [astArg, type] :
       llvm::zip(subroutine.getArguments(),
                 lowering->op.getFunctionType().getInputs())) {
    auto loc = convertLocation(astArg->location);
    auto blockArg = block.addArgument(type, loc);

    if (isa<moore::RefType>(type)) {
      valueSymbols.insert(astArg, blockArg);
    } else {
      // Convert the body of the function.
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&block);

      auto shadowArg = moore::VariableOp::create(
          builder, loc, moore::RefType::get(cast<moore::UnpackedType>(type)),
          StringAttr{}, blockArg);
      valueSymbols.insert(astArg, shadowArg);
      argVariables.push_back(shadowArg);
    }
  }

  // Convert the body of the function.
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(&block);

  Value returnVar;
  if (subroutine.returnValVar) {
    auto type = convertType(*subroutine.returnValVar->getDeclaredType());
    if (!type)
      return failure();
    returnVar = moore::VariableOp::create(
        builder, lowering->op.getLoc(),
        moore::RefType::get(cast<moore::UnpackedType>(type)), StringAttr{},
        Value{});
    valueSymbols.insert(subroutine.returnValVar, returnVar);
  }

  if (failed(convertStatement(subroutine.getBody())))
    return failure();

  // If there was no explicit return statement provided by the user, insert a
  // default one.
  if (builder.getBlock()) {
    if (returnVar && !subroutine.getReturnType().isVoid()) {
      Value read =
          moore::ReadOp::create(builder, returnVar.getLoc(), returnVar);
      mlir::func::ReturnOp::create(builder, lowering->op.getLoc(), read);
    } else {
      mlir::func::ReturnOp::create(builder, lowering->op.getLoc(),
                                   ValueRange{});
    }
  }
  if (returnVar && returnVar.use_empty())
    returnVar.getDefiningOp()->erase();

  for (auto var : argVariables) {
    if (llvm::all_of(var->getUsers(),
                     [](auto *user) { return isa<moore::ReadOp>(user); })) {
      for (auto *user : llvm::make_early_inc_range(var->getUsers())) {
        user->getResult(0).replaceAllUsesWith(var.getInitial());
        user->erase();
      }
      var->erase();
    }
  }
  return success();
}
