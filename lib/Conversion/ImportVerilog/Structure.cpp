//===- Structure.cpp - Slang hierarchy conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/SystemSubroutine.h"
#include "slang/ast/symbols/ClassSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
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

static Type getInterfaceSignalType(Type type) {
  if (auto intTy = dyn_cast<moore::IntType>(type)) {
    unsigned width = intTy.getBitSize().value_or(1);
    if (width == 0)
      width = 1;
    auto widthAttr = IntegerAttr::get(
        IntegerType::get(type.getContext(), 32), APInt(32, width));
    Type elemTy = hw::IntType::get(widthAttr);
    if (intTy.getDomain() == moore::Domain::FourValued) {
      SmallVector<hw::StructType::FieldInfo> fields;
      fields.push_back(
          {StringAttr::get(type.getContext(), "value"), elemTy});
      fields.push_back(
          {StringAttr::get(type.getContext(), "unknown"), elemTy});
      return hw::StructType::get(type.getContext(), fields);
    }
    return elemTy;
  }
  return type;
}

static void guessNamespacePrefix(const slang::ast::Symbol &symbol,
                                 SmallString<64> &prefix) {
  if (symbol.kind != slang::ast::SymbolKind::Package)
    return;
  if (const auto *parent = symbol.getParentScope())
    guessNamespacePrefix(parent->asSymbol(), prefix);
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
static LogicalResult convertClassDecl(Context &context,
                                     const slang::ast::ClassType &cls,
                                     Location loc) {
  context.getOrAssignClassId(cls);
  if (const auto *baseType = cls.getBaseClass())
    if (const auto *baseCls = baseType->as_if<slang::ast::ClassType>())
      context.getOrAssignClassId(*baseCls);

  for (auto &prop : cls.membersOfType<slang::ast::ClassPropertySymbol>())
    context.getOrAssignClassFieldId(prop);

  for (auto &method : cls.membersOfType<slang::ast::SubroutineSymbol>())
    if (failed(context.convertFunction(method)))
      return failure();

  return success();
}

struct RootVisitor : public BaseVisitor {
  using BaseVisitor::BaseVisitor;
  using BaseVisitor::visit;

  LogicalResult visit(const slang::ast::ClassType &cls) {
    if (context.options.allowClassStubs)
      return success();
    return convertClassDecl(context, cls, loc);
  }
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

    auto it = context.orderedRootOps.upper_bound(varNode.location);
    if (it == context.orderedRootOps.end())
      builder.setInsertionPointToEnd(context.intoModuleOp.getBody());
    else
      builder.setInsertionPoint(it->second);

    Value initial;
    if (const auto *init = varNode.getInitializer()) {
      initial = context.convertRvalueExpression(*init, loweredType);
      if (!initial)
        return failure();
    }

    // Treat top-level string variables as compile-time constants. sv-tests
    // uses these as labels in UVM macros; lowering them to storage currently
    // introduces LLHD vars that are not legalizable through the Arc pipeline.
    if (isa<moore::StringType>(loweredType)) {
      if (!initial) {
        mlir::emitError(loc,
                        "top-level string variable without initializer is "
                        "unsupported");
        return failure();
      }
      if (auto *defOp = initial.getDefiningOp())
        context.orderedRootOps.insert(it, {varNode.location, defOp});
      context.valueSymbols.insert(&varNode, initial);
      return success();
    }

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

  LogicalResult visit(const slang::ast::ClassType &node) {
    if (context.options.allowClassStubs)
      return success();
    return convertClassDecl(context, node, loc);
  }
  LogicalResult visit(const slang::ast::GenericClassDefSymbol &node) {
    if (context.options.allowClassStubs)
      return success();
    mlir::emitError(loc, "unsupported package member: generic class `")
        << node.name << "`";
    return failure();
  }

  // Handle functions and tasks.
  LogicalResult visit(const slang::ast::SubroutineSymbol &subroutine) {
    if (const auto *parentScope = subroutine.getParentScope()) {
      const auto &parentSym = parentScope->asSymbol();
      if (parentSym.kind == slang::ast::SymbolKind::ClassType ||
          parentSym.kind == slang::ast::SymbolKind::GenericClassDef) {
        if (context.options.allowClassStubs)
          return success();
        mlir::emitError(loc, "unsupported package member: class method `")
            << subroutine.name << "`";
        return failure();
      }
    }
    return context.convertFunction(subroutine);
  }

  // Handle package-scope variables.
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

  LogicalResult visit(const slang::ast::ClassType &cls) {
    if (context.options.allowClassStubs)
      return success();
    return convertClassDecl(context, cls, loc);
  }
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

      // Lower interface port bindings (e.g. `input_if in(clk);`) by wiring the
      // provided expressions into the corresponding interface signals. Slang
      // exposes these as regular port connections on the interface instance.
      //
      // Note: This is intentionally conservative and focuses on the most
      // common patterns (clock/reset inputs, and basic output wiring). Inout
      // and ref ports are currently ignored.
      for (const auto *con : instNode.getPortConnections()) {
        const auto *expr = con->getExpression();
        if (!expr)
          continue;

        // Unpack the `<expr> = EmptyArgument` pattern emitted by Slang for
        // output and inout ports.
        if (const auto *assign = expr->as_if<AssignmentExpression>())
          expr = &assign->left();

        auto *port = con->port.as_if<PortSymbol>();
        if (!port)
          continue;

        const auto *internalValue =
            port->internalSymbol
                ? port->internalSymbol->as_if<slang::ast::ValueSymbol>()
                : nullptr;
        if (!internalValue) {
          mlir::emitError(loc)
              << "unsupported interface port `" << port->name
              << "` (missing internal value symbol)";
          return failure();
        }

        auto memberLoc = context.convertLocation(internalValue->location);
        auto signalAttr = context.lookupInterfaceSignal(*internalValue, memberLoc);
        if (failed(signalAttr))
          return failure();

        auto targetType = context.convertType(port->getType());
        if (!targetType)
          return failure();
        auto signalType = getInterfaceSignalType(targetType);
        if (!signalType)
          return failure();

        switch (port->direction) {
        case ArgumentDirection::In: {
          Value rhs = context.convertRvalueExpression(*expr, targetType);
          if (!rhs)
            return failure();
          rhs = context.materializeConversion(signalType, rhs, /*isSigned=*/false,
                                              rhs.getLoc());
          if (!rhs)
            return failure();

          builder.create<sv::AssignInterfaceSignalOp>(
              memberLoc, inst.getResult(), *signalAttr, rhs);
          break;
        }
        case ArgumentDirection::Out: {
          // Bind the interface signal to the provided lvalue.
          Value lvalue = context.convertLvalueExpression(*expr);
          if (!lvalue)
            return failure();
          auto dstType = cast<moore::RefType>(lvalue.getType()).getNestedType();
          Value read = builder.create<sv::ReadInterfaceSignalOp>(
              memberLoc, signalType, inst.getResult(), *signalAttr);
          read = context.materializeConversion(dstType, read,
                                               /*isSigned=*/false, memberLoc);
          if (!read)
            return failure();
          moore::ContinuousAssignOp::create(builder, memberLoc, lvalue, read);
          break;
        }
        default:
          // Best-effort: ignore inout/ref ports for now.
          break;
        }
      }

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

    struct InterfaceSignalAssign {
      Value ifaceValue;
      FlatSymbolRefAttr signalAttr;
      Type signalType;
      Location memberLoc;
    };
    SmallDenseMap<const PortSymbol *, InterfaceSignalAssign> ifaceSignalAssigns;

    auto resolveIfaceHandleFromHier =
        [&](const slang::ast::HierarchicalValueExpression &hier) -> Value {
      auto lookUpSymbol = [&](const slang::ast::Symbol *symbol) -> Value {
        if (!symbol)
          return Value();
        if (auto *ifacePort =
                symbol->as_if<slang::ast::InterfacePortSymbol>()) {
          if (auto value = context.valueSymbols.lookup(ifacePort);
              value && (isa<sv::InterfaceType>(value.getType()) ||
                        isa<sv::ModportType>(value.getType())))
            return value;
        }
        if (auto *instSym = symbol->as_if<slang::ast::InstanceSymbol>()) {
          if (auto value = context.valueSymbols.lookup(instSym);
              value && (isa<sv::InterfaceType>(value.getType()) ||
                        isa<sv::ModportType>(value.getType())))
            return value;
        }
        if (auto *valueSym = symbol->as_if<slang::ast::ValueSymbol>()) {
          if (auto value = context.valueSymbols.lookup(valueSym);
              value && (isa<sv::InterfaceType>(value.getType()) ||
                        isa<sv::ModportType>(value.getType())))
            return value;
        }
        return Value();
      };

      for (const auto &element : hier.ref.path)
        if (auto value = lookUpSymbol(element.symbol))
          return value;

      if (auto value = lookUpSymbol(hier.ref.target))
        return value;

      return Value();
    };

    auto recordInterfaceSignalAssign =
        [&](const slang::ast::Expression &expr, const PortSymbol *port)
        -> LogicalResult {
      auto *hier = expr.as_if<slang::ast::HierarchicalValueExpression>();
      if (!hier)
        return failure();

      bool isViaInterface = hier->ref.isViaIfacePort();
      if (!isViaInterface) {
        for (const auto &element : hier->ref.path) {
          if (auto *instSym =
                  element.symbol->as_if<slang::ast::InstanceSymbol>()) {
            if (instSym->body.getDefinition().definitionKind ==
                slang::ast::DefinitionKind::Interface) {
              isViaInterface = true;
              break;
            }
          }
        }
      }
      if (!isViaInterface)
        return failure();

      auto ifaceValue = resolveIfaceHandleFromHier(*hier);
      if (!ifaceValue)
        return failure();

      auto *memberSym = hier->symbol.as_if<slang::ast::ValueSymbol>();
      if (!memberSym) {
        mlir::emitError(loc)
            << "interface member `" << hier->symbol.name
            << "` is not a value symbol";
        return failure();
      }

      auto memberLoc = context.convertLocation(hier->symbol.location);
      auto signalAttr = context.lookupInterfaceSignal(*memberSym, memberLoc);
      if (failed(signalAttr))
        return failure();

      auto targetType = context.convertType(*hier->type);
      if (!targetType)
        return failure();
      auto signalType = getInterfaceSignalType(targetType);
      if (!signalType)
        return failure();

      ifaceSignalAssigns.insert(
          {port,
           InterfaceSignalAssign{ifaceValue, *signalAttr, signalType,
                                 memberLoc}});
      return success();
    };

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
        if (auto *existingPort =
                moduleLowering->portsBySyntaxNode.lookup(con->port.getSyntax()))
          port = existingPort;
        Value value;
        if (port->direction == ArgumentDirection::In) {
          value = context.convertRvalueExpression(*expr);
          if (!value)
            return failure();
          portValues.insert({port, value});
          continue;
        }
        // Allow connecting a module output port to an interface signal.
        if (port->direction == ArgumentDirection::Out &&
            succeeded(recordInterfaceSignalAssign(*expr, port)))
          continue;
        value = context.convertLvalueExpression(*expr);
        if (!value)
          return failure();
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
    SmallVector<const PortSymbol *> outputPorts;
    inputValues.reserve(moduleType.getNumInputs());
    outputValues.reserve(moduleType.getNumOutputs());
    outputPorts.reserve(moduleType.getNumOutputs());

    for (auto &info : moduleLowering->portInfos) {
      if (info.kind == ModuleLowering::ModulePortInfo::Kind::Data) {
        auto &port = moduleLowering->ports[info.index];
        auto value = portValues.lookup(&port.ast);
        if (port.ast.direction == ArgumentDirection::Out) {
          outputValues.push_back(value);
          outputPorts.push_back(&port.ast);
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
    for (auto [idx, output] : llvm::enumerate(inst.getOutputs())) {
      auto *port = outputPorts[idx];
      auto lvalue = outputValues[idx];
      if (lvalue) {
        Value rvalue = output;
        auto dstType = cast<moore::RefType>(lvalue.getType()).getNestedType();
        // TODO: This should honor signedness in the conversion.
        rvalue = context.materializeConversion(dstType, rvalue, false, loc);
        moore::ContinuousAssignOp::create(builder, loc, lvalue, rvalue);
        continue;
      }
      auto ifaceAssignIt = ifaceSignalAssigns.find(port);
      if (ifaceAssignIt != ifaceSignalAssigns.end()) {
        const auto &assign = ifaceAssignIt->second;
        Value rvalue = output;
        rvalue = context.materializeConversion(assign.signalType, rvalue,
                                               /*isSigned=*/false, loc);
        if (!rvalue)
          return failure();
        builder.create<sv::AssignInterfaceSignalOp>(
            assign.memberLoc, assign.ifaceValue, assign.signalAttr, rvalue);
      }
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
      // Support assignments to signals within interface instances.
      for (const auto &element : hier->ref.path) {
        if (auto *instSym =
                element.symbol->as_if<slang::ast::InstanceSymbol>()) {
          if (instSym->body.getDefinition().definitionKind !=
              slang::ast::DefinitionKind::Interface)
            continue;
          auto assigned =
              context.assignInterfaceMember(expr.left(), expr.right(), loc);
          if (failed(assigned))
            return failure();
          return success();
        }
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
    auto &entryBlock = procOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&entryBlock);
    Context::ValueSymbolScope scope(context.valueSymbols);

    // Some SV frontends lower module-scope concurrent assertions into
    // clocked procedural blocks. Provide best-effort support for sampled-value
    // assertion system calls (`$past`, `$rose`, ...) when they appear in such
    // procedural contexts by synthesizing per-procedure state.
    DenseMap<const slang::ast::Expression *, Value> prevVars;
    DenseMap<const slang::ast::Expression *, Value> curValues;
    unsigned nextPrevId = 0;
    Value havePastVar;
    unsigned scopeId = context.nextAssertionCallScopeId++;
    auto prefix = ("__svtests_pa" + Twine(scopeId)).str();

    auto toMatchingIntType = [&](Value value,
                                 moore::IntType targetType) -> Value {
      if (!value)
        return {};
      auto srcType = dyn_cast<moore::IntType>(value.getType());
      if (!srcType)
        return {};

      Value casted = value;
      if (srcType.getDomain() != targetType.getDomain()) {
        if (srcType.getDomain() == Domain::TwoValued &&
            targetType.getDomain() == Domain::FourValued) {
          casted = moore::IntToLogicOp::create(builder, value.getLoc(), casted);
        } else if (srcType.getDomain() == Domain::FourValued &&
                   targetType.getDomain() == Domain::TwoValued) {
          casted =
              moore::LogicToIntOp::create(builder, value.getLoc(), casted);
        } else {
          return {};
        }
      }

      // Replicate a 1-bit guard to match the target width, if needed.
      auto srcBits = srcType.getBitSize();
      auto dstBits = targetType.getBitSize();
      if (!srcBits || !dstBits)
        return {};
      if (*srcBits != *dstBits) {
        if (*srcBits != 1)
          return {};
        casted = moore::ReplicateOp::create(builder, value.getLoc(), targetType,
                                            casted);
      }
      return casted;
    };

    auto ensureHavePastVar = [&](Location declLoc) -> Value {
      if (havePastVar)
        return havePastVar;

      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(&entryBlock);
      auto havePastTy =
          moore::IntType::get(context.getContext(), 1, Domain::TwoValued);
      Value init0 = moore::ConstantOp::create(builder, declLoc, havePastTy, 0);
      havePastVar = moore::VariableOp::create(
          builder, declLoc,
          moore::RefType::get(cast<moore::UnpackedType>(havePastTy)),
          builder.getStringAttr(prefix + "_has_past"), init0);
      return havePastVar;
    };

    auto getOrCreatePrevVar =
        [&](const slang::ast::Expression *argExpr, moore::IntType intTy,
            Location declLoc) -> Value {
      if (!argExpr)
        return {};
      if (Value existing = prevVars.lookup(argExpr))
        return existing;

      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(&entryBlock);
      Value init0 = moore::ConstantOp::create(builder, declLoc, intTy, 0);
      auto name = builder.getStringAttr(
          (Twine(prefix) + "_prev" + Twine(nextPrevId++)).str());
      Value var = moore::VariableOp::create(
          builder, declLoc,
          moore::RefType::get(cast<moore::UnpackedType>(intTy)), name, init0);
      prevVars.insert({argExpr, var});
      return var;
    };

    auto savedOverride = context.assertionCallOverride;
    context.assertionCallOverride =
        [&](const slang::ast::CallExpression &expr,
            const slang::ast::CallExpression::SystemCallInfo &info,
            Location callLoc) -> Value {
      const auto &subroutine = *info.subroutine;
      auto args = expr.arguments();
      if (args.size() != 1 || !args[0])
        return {};

      Value argVal = context.convertRvalueExpression(*args[0]);
      if (!argVal)
        return {};
      auto intTy = dyn_cast<moore::IntType>(argVal.getType());
      if (!intTy)
        return {};

      const slang::ast::Expression *argExpr = args[0];
      Value prevVar = getOrCreatePrevVar(argExpr, intTy, callLoc);
      if (!prevVar)
        return {};
      if (!curValues.count(argExpr))
        curValues.insert({argExpr, argVal});

      Value havePastRef = ensureHavePastVar(callLoc);
      if (!havePastRef)
        return {};
      Value havePast = moore::ReadOp::create(builder, callLoc, havePastRef);
      Value prevVal = moore::ReadOp::create(builder, callLoc, prevVar);

      auto guardAnd = [&](Value guard, Value value) -> Value {
        auto valueTy = dyn_cast<moore::IntType>(value.getType());
        auto guardTy = dyn_cast<moore::IntType>(guard.getType());
        if (!valueTy || !guardTy)
          return Value{};
        Value guardCast = toMatchingIntType(guard, valueTy);
        if (!guardCast)
          return {};
        return moore::AndOp::create(builder, callLoc, guardCast, value);
      };

      if (subroutine.name == "$past") {
        // Best-effort: `$past(x)` is `x[-1]` if a past sample exists, else 0.
        return guardAnd(havePast, prevVal);
      }

      if (subroutine.name == "$stable") {
        Value eq = moore::EqOp::create(builder, callLoc, argVal, prevVal);
        auto eqTy = cast<moore::IntType>(eq.getType());
        Value havePastBit = toMatchingIntType(havePast, eqTy);
        if (!havePastBit)
          return {};
        return moore::AndOp::create(builder, callLoc, havePastBit, eq);
      }

      if (subroutine.name == "$changed") {
        Value eq = moore::EqOp::create(builder, callLoc, argVal, prevVal);
        auto eqTy = cast<moore::IntType>(eq.getType());
        Value havePastBit = toMatchingIntType(havePast, eqTy);
        if (!havePastBit)
          return {};
        Value stable = moore::AndOp::create(builder, callLoc, havePastBit, eq);
        return moore::NotOp::create(builder, callLoc, stable);
      }

      // $rose/$fell are defined on single-bit arguments.
      auto width = intTy.getBitSize();
      if (!width || *width != 1)
        return {};

      if (subroutine.name == "$rose") {
        Value notPrev = moore::NotOp::create(builder, callLoc, prevVal);
        Value core = moore::AndOp::create(builder, callLoc, notPrev, argVal);
        return guardAnd(havePast, core);
      }

      if (subroutine.name == "$fell") {
        Value notCur = moore::NotOp::create(builder, callLoc, argVal);
        Value core = moore::AndOp::create(builder, callLoc, prevVal, notCur);
        return guardAnd(havePast, core);
      }

      return {};
    };
    auto restoreOverride = llvm::make_scope_exit(
        [&] { context.assertionCallOverride = savedOverride; });

    if (failed(context.convertStatement(body)))
      return failure();
    if (builder.getBlock()) {
      if (havePastVar) {
        for (auto [expr, cur] : curValues) {
          Value prevVar = prevVars.lookup(expr);
          if (!prevVar)
            continue;
          moore::BlockingAssignOp::create(builder, loc, prevVar, cur);
        }
        auto havePastTy =
            moore::IntType::get(context.getContext(), 1, Domain::TwoValued);
        Value one = moore::ConstantOp::create(builder, loc, havePastTy, 1);
        moore::BlockingAssignOp::create(builder, loc, havePastVar, one);
      }
      moore::ReturnOp::create(builder, loc);
    }
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

  if (failed(finalizeUvmShims()))
    return failure();

  return success();
}

LogicalResult Context::finalizeUvmShims() {
  // UVM shim bring-up: we generate a minimal phase scheduler shim when
  // lowering `uvm_root::run_test`. At that point the set of instantiated
  // component classes may not be known yet, so the generated dispatch stubs can
  // be empty. Rebuild the phase-dispatch functions once we've converted the
  // whole compilation and have assigned class IDs.
  if (!intoModuleOp.lookupSymbol<mlir::func::FuncOp>("circt_uvm_run_all_phases"))
    return success();

  auto isUvmPkgScope = [](const slang::ast::Scope *scope) -> bool {
    if (!scope)
      return false;
    const auto &sym = scope->asSymbol();
    return sym.kind == slang::ast::SymbolKind::Package && sym.name == "uvm_pkg";
  };

  auto isUvmLibraryMethod =
      [&](const slang::ast::SubroutineSymbol &method) -> bool {
    const auto *ownerScope = method.getParentScope();
    if (!ownerScope)
      return false;
    const auto &ownerSym = ownerScope->asSymbol();
    if (ownerSym.kind != slang::ast::SymbolKind::ClassType)
      return false;

    auto isUvmName = [&](llvm::StringRef name) -> bool {
      return name.starts_with("uvm_");
    };

    bool ownerIsUvm = isUvmName(
        llvm::StringRef(ownerSym.name.data(), ownerSym.name.size()));
    if (!ownerIsUvm)
      if (auto *ownerCls = ownerSym.as_if<slang::ast::ClassType>())
        if (ownerCls->genericClass)
          ownerIsUvm = isUvmName(llvm::StringRef(
              ownerCls->genericClass->name.data(),
              ownerCls->genericClass->name.size()));
    if (!ownerIsUvm)
      return false;

    if (isUvmPkgScope(ownerSym.getParentScope()))
      return true;
    if (auto *ownerCls = ownerSym.as_if<slang::ast::ClassType>())
      if (ownerCls->genericClass &&
          isUvmPkgScope(ownerCls->genericClass->getParentScope()))
        return true;

    // Name-based fallback: treat `uvm_*` methods as UVM even if the package
    // scope isn't directly visible (e.g. generated specializations).
    return true;
  };

  auto isUvmDriverClass = [&](const slang::ast::ClassType &cls) -> bool {
    if (cls.name == "uvm_driver")
      return true;
    if (cls.genericClass && cls.genericClass->name == "uvm_driver")
      return true;
    return false;
  };

  auto findUvmDriverBase =
      [&](const slang::ast::ClassType &cls) -> const slang::ast::ClassType * {
    const slang::ast::Type *base = cls.getBaseClass();
    while (base) {
      const auto *baseCls =
          base->getCanonicalType().as_if<slang::ast::ClassType>();
      if (!baseCls)
        break;
      if (isUvmDriverClass(*baseCls)) {
        // Prefer confirming the origin via the uvm_pkg scope, but accept
        // name-based matches as a fallback (the stub UVM library can
        // materialize generic-class specializations in non-package scopes).
        if (isUvmPkgScope(baseCls->getParentScope()))
          return baseCls;
        if (baseCls->genericClass &&
            isUvmPkgScope(baseCls->genericClass->getParentScope()))
          return baseCls;
        return baseCls;
      }
      base = baseCls->getBaseClass();
    }
    return nullptr;
  };

  auto ensureUvmDriverEndOfElabShim =
      [&](const slang::ast::ClassType &uvmDriverCls, FunctionType fnTy,
          Location loc) -> mlir::func::FuncOp {
    StringRef shimName = "__circt_uvm_shim_uvm_driver_end_of_elaboration_phase";
    if (auto existing =
            intoModuleOp.lookupSymbol<mlir::func::FuncOp>(shimName))
      return existing;

    auto i32Ty = moore::IntType::get(getContext(), /*width=*/32,
                                     moore::Domain::TwoValued);

    const slang::ast::ClassPropertySymbol *seqItemPortProp = nullptr;
    for (auto &prop :
         uvmDriverCls.membersOfType<slang::ast::ClassPropertySymbol>()) {
      if (prop.name == "seq_item_port") {
        seqItemPortProp = &prop;
        break;
      }
    }
    if (!seqItemPortProp)
      return mlir::func::FuncOp();

    int32_t fieldId = getOrAssignClassFieldId(*seqItemPortProp);

    auto getOrCreateExternFunc = [&](StringRef name, FunctionType type) {
      if (auto existing = intoModuleOp.lookupSymbol<mlir::func::FuncOp>(name)) {
        if (existing.getFunctionType() != type) {
          mlir::emitError(loc, "conflicting declarations for `")
              << name << "`";
          return mlir::func::FuncOp();
        }
        return existing;
      }

      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(intoModuleOp.getBody());
      getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
      auto fn = mlir::func::FuncOp::create(builder, loc, name, type);
      fn.setPrivate();
      symbolTable.insert(fn);
      return fn;
    };

    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(intoModuleOp.getBody());
    getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
    auto shimFn = mlir::func::FuncOp::create(builder, loc, shimName, fnTy);
    shimFn.setPrivate();
    symbolTable.insert(shimFn);

    auto &entry = shimFn.getBody().emplaceBlock();
    for (auto ty : fnTy.getInputs())
      entry.addArgument(ty, loc);

    OpBuilder b(getContext());
    b.setInsertionPointToStart(&entry);
    Value selfArg = entry.getArgument(0);

    Value fieldIdVal = moore::ConstantOp::create(b, loc, i32Ty, fieldId,
                                                 /*isSigned=*/true);

    auto getFnTy = FunctionType::get(getContext(), {i32Ty, i32Ty}, {i32Ty});
    auto getFn = getOrCreateExternFunc("circt_sv_class_get_i32", getFnTy);
    if (!getFn)
      return mlir::func::FuncOp();
    Value portHandle = mlir::func::CallOp::create(
                           b, loc, getFn, ValueRange{selfArg, fieldIdVal})
                           .getResult(0);

    auto countFnTy = FunctionType::get(getContext(), {i32Ty}, {i32Ty});
    auto countFn = getOrCreateExternFunc("circt_uvm_port_conn_count", countFnTy);
    if (!countFn)
      return mlir::func::FuncOp();
    Value count = mlir::func::CallOp::create(b, loc, countFn,
                                             ValueRange{portHandle})
                      .getResult(0);

    Value one =
        moore::ConstantOp::create(b, loc, i32Ty, /*value=*/1, /*isSigned=*/true);
    Value lt = moore::UltOp::create(b, loc, count, one);
    Value cond = moore::ToBuiltinBoolOp::create(b, loc, lt);

    Block &warnBlock = shimFn.getBody().emplaceBlock();
    Block &doneBlock = shimFn.getBody().emplaceBlock();
    mlir::cf::CondBranchOp::create(b, loc, cond, &warnBlock, ValueRange{},
                                   &doneBlock, ValueRange{});

    auto makeString = [&](StringRef value) -> Value {
      auto fmt =
          moore::FormatLiteralOp::create(b, loc, b.getStringAttr(value));
      return moore::FormatStringToStringOp::create(b, loc, fmt);
    };

    b.setInsertionPointToStart(&warnBlock);
    Value id = makeString("DRVCONNECT");
    Value message = makeString(
        "the driver is not connected to a sequencer via the standard "
        "mechanisms enabled by connect()");
    Value sevVal = moore::ConstantOp::create(b, loc, i32Ty, /*value=*/1,
                                             /*isSigned=*/true);
    auto reportFnTy = FunctionType::get(
        getContext(), {i32Ty, i32Ty, id.getType(), message.getType()}, {});
    auto reportFn = getOrCreateExternFunc("circt_uvm_report", reportFnTy);
    if (!reportFn)
      return mlir::func::FuncOp();
    mlir::func::CallOp::create(b, loc, reportFn,
                               ValueRange{selfArg, sevVal, id, message});
    mlir::cf::BranchOp::create(b, loc, &doneBlock, ValueRange{});

    b.setInsertionPointToStart(&doneBlock);
    mlir::func::ReturnOp::create(b, loc);
    return shimFn;
  };

	  auto rebuildPhaseDispatch = [&](StringRef methodName,
	                                  StringRef fnName) -> LogicalResult {
	    Location loc = UnknownLoc::get(getContext());

    mlir::func::FuncOp dispatchFn =
        intoModuleOp.lookupSymbol<mlir::func::FuncOp>(fnName);
    if (!dispatchFn) {
      auto i32Ty = moore::IntType::get(getContext(), /*width=*/32,
                                       moore::Domain::TwoValued);
      auto fnTy = FunctionType::get(getContext(), {i32Ty, i32Ty}, {});

      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(intoModuleOp.getBody());
      getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
      dispatchFn = mlir::func::FuncOp::create(builder, loc, fnName, fnTy);
      dispatchFn.setPrivate();
      symbolTable.insert(dispatchFn);
    }

    auto fnTy = dispatchFn.getFunctionType();
    if (fnTy.getNumInputs() < 1)
      return success();

	    SmallVector<std::pair<const slang::ast::ClassType *, int32_t>>
	        classIdSnapshot;
	    classIdSnapshot.reserve(classIds.size());
	    for (auto &entry : classIds)
	      classIdSnapshot.push_back({entry.first, entry.second});

	    SmallVector<std::pair<int32_t, mlir::func::FuncOp>> impls;
	    for (auto [cls, classId] : classIdSnapshot) {
	      if (!cls)
	        continue;
	      llvm::StringRef clsName(cls->name.data(), cls->name.size());
	      if (clsName.starts_with("uvm_"))
        continue;
      const slang::ast::SubroutineSymbol *impl = nullptr;
      const slang::ast::SubroutineSymbol *inheritedImpl = nullptr;
      for (auto &method : cls->membersOfType<slang::ast::SubroutineSymbol>()) {
        if (llvm::StringRef(method.name.data(), method.name.size()) !=
            methodName)
          continue;
        if (isUvmLibraryMethod(method))
          continue;

        const auto *ownerScope = method.getParentScope();
        if (ownerScope && &ownerScope->asSymbol() == cls) {
          impl = &method;
          break;
        }
        if (!inheritedImpl)
          inheritedImpl = &method;
      }
      if (!impl)
        impl = inheritedImpl;
      if (!impl) {
        if (methodName == "end_of_elaboration_phase") {
          if (const auto *uvmDriverBase = findUvmDriverBase(*cls)) {
            if (auto shimFn =
                    ensureUvmDriverEndOfElabShim(*uvmDriverBase, fnTy, loc))
              impls.push_back({classId, shimFn});
          }
        }
        continue;
      }
      if (failed(convertFunction(*impl)))
        continue;
      auto *implLowering = declareFunction(*impl);
      if (!implLowering || !implLowering->op)
        continue;
      if (implLowering->op.getFunctionType() != fnTy)
        continue;
      impls.push_back({classId, implLowering->op});
    }
    llvm::sort(impls, [](const auto &a, const auto &b) {
      return a.first < b.first;
    });

    dispatchFn.getBody().getBlocks().clear();

    auto &entry = dispatchFn.getBody().emplaceBlock();
    for (auto ty : fnTy.getInputs())
      entry.addArgument(ty, loc);

    OpBuilder b(getContext());
    b.setInsertionPointToStart(&entry);
    if (impls.empty()) {
      mlir::func::ReturnOp::create(b, loc);
      return success();
    }

    Value thisArg = entry.getArgument(0);
    auto getTypeFnTy = FunctionType::get(getContext(), {thisArg.getType()},
                                         {thisArg.getType()});
    auto getTypeFn = intoModuleOp.lookupSymbol<mlir::func::FuncOp>(
        "circt_sv_class_get_type");
    if (!getTypeFn || getTypeFn.getFunctionType() != getTypeFnTy) {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(intoModuleOp.getBody());
      getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
      getTypeFn = mlir::func::FuncOp::create(builder, loc,
                                             "circt_sv_class_get_type",
                                             getTypeFnTy);
      getTypeFn.setPrivate();
      symbolTable.insert(getTypeFn);
    }

    Value dynType =
        mlir::func::CallOp::create(b, loc, getTypeFn, {thisArg}).getResult(0);

    Block *defaultBlock = &dispatchFn.getBody().emplaceBlock();
    SmallVector<Block *> checkBlocks;
    SmallVector<Block *> caseBlocks;
    checkBlocks.reserve(impls.size());
    caseBlocks.reserve(impls.size());
    for (size_t i = 0, e = impls.size(); i < e; ++i) {
      checkBlocks.push_back(&dispatchFn.getBody().emplaceBlock());
      caseBlocks.push_back(&dispatchFn.getBody().emplaceBlock());
    }

    b.setInsertionPointToEnd(&entry);
    mlir::cf::BranchOp::create(b, loc,
                               checkBlocks.empty() ? defaultBlock
                                                   : checkBlocks.front());

    b.setInsertionPointToStart(defaultBlock);
    mlir::func::ReturnOp::create(b, loc);

    auto cmpTy = moore::IntType::get(
        getContext(), /*width=*/32,
        cast<moore::IntType>(dynType.getType()).getDomain());

    for (size_t i = 0, e = impls.size(); i < e; ++i) {
      auto [classId, implFn] = impls[i];
      Block *next = (i + 1 < e) ? checkBlocks[i + 1] : defaultBlock;

      b.setInsertionPointToStart(checkBlocks[i]);
      Value classIdVal = moore::ConstantOp::create(b, loc, cmpTy, classId,
                                                   /*isSigned=*/true);
      Value eq = b.createOrFold<moore::EqOp>(loc, dynType, classIdVal);
      eq = b.createOrFold<moore::BoolCastOp>(loc, eq);
      Value cond = moore::ToBuiltinBoolOp::create(b, loc, eq);
      mlir::cf::CondBranchOp::create(b, loc, cond, caseBlocks[i], next);

      b.setInsertionPointToStart(caseBlocks[i]);
      mlir::func::CallOp::create(b, loc, implFn, entry.getArguments());
      mlir::func::ReturnOp::create(b, loc);
    }

    return success();
  };

  if (failed(
          rebuildPhaseDispatch("build_phase", "__circt_uvm_dispatch_build_phase")))
    return failure();
  if (failed(rebuildPhaseDispatch("connect_phase",
                                  "__circt_uvm_dispatch_connect_phase")))
    return failure();
  if (failed(rebuildPhaseDispatch(
          "end_of_elaboration_phase",
          "__circt_uvm_dispatch_end_of_elaboration_phase")))
    return failure();
  if (failed(rebuildPhaseDispatch(
          "start_of_simulation_phase",
          "__circt_uvm_dispatch_start_of_simulation_phase")))
    return failure();
  if (failed(
          rebuildPhaseDispatch("run_phase", "__circt_uvm_dispatch_run_phase")))
    return failure();
  if (failed(rebuildPhaseDispatch("extract_phase",
                                  "__circt_uvm_dispatch_extract_phase")))
    return failure();
  if (failed(rebuildPhaseDispatch("check_phase",
                                  "__circt_uvm_dispatch_check_phase")))
    return failure();
  if (failed(rebuildPhaseDispatch("report_phase",
                                  "__circt_uvm_dispatch_report_phase")))
    return failure();

  // Rebuild the minimal analysis-port dispatch shim used by the UVM bring-up
  // runtime. The `uvm_analysis_port::write` lowering can be encountered before
  // we've assigned IDs / converted all user classes (e.g. monitors defined
  // before scoreboards). In that case, the initial dispatch stub may be empty
  // and later writes become no-ops.
  if (intoModuleOp.lookupSymbol<mlir::func::FuncOp>(
          "__circt_uvm_analysis_port_write") ||
      intoModuleOp.lookupSymbol<mlir::func::FuncOp>(
          "__circt_uvm_dispatch_analysis_write")) {
    auto i32Ty = moore::IntType::get(getContext(), /*width=*/32,
                                     moore::Domain::TwoValued);
    auto fnTy = FunctionType::get(getContext(), {i32Ty, i32Ty}, {});

    int32_t analysisImpId = 0;
    for (auto [cls, classId] : classIds) {
      if (!cls)
        continue;
      llvm::StringRef clsName(cls->name.data(), cls->name.size());
      if (clsName == "uvm_analysis_imp") {
        analysisImpId = classId;
        break;
      }
    }

	    auto rebuildAnalysisDispatch = [&]() -> LogicalResult {
	      Location loc = UnknownLoc::get(getContext());

	      mlir::func::FuncOp dispatchFn =
	          intoModuleOp.lookupSymbol<mlir::func::FuncOp>(
	              "__circt_uvm_dispatch_analysis_write");
	      if (!dispatchFn) {
	        OpBuilder::InsertionGuard g(builder);
	        builder.setInsertionPointToStart(intoModuleOp.getBody());
	        getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
	        dispatchFn = mlir::func::FuncOp::create(
	            builder, loc, "__circt_uvm_dispatch_analysis_write", fnTy);
	        dispatchFn.setPrivate();
	        symbolTable.insert(dispatchFn);
	      }

	      // Avoid deleting existing blocks: the initial dispatch stub may already
	      // contain a CFG, and wiping it here can invalidate in-flight builder
	      // state. Instead, overwrite the entry terminator to jump to a freshly
	      // built dispatcher, leaving the old blocks unreachable.
	      Block *entry = dispatchFn.getBody().empty()
	                         ? &dispatchFn.getBody().emplaceBlock()
	                         : &dispatchFn.getBody().front();
	      if (entry->getNumArguments() == 0)
	        for (auto ty : fnTy.getInputs())
	          entry->addArgument(ty, loc);

	      OpBuilder b(getContext());
	      // Drop the existing terminator (if any) so we can append the rebuilt
	      // dispatcher logic to the entry block.
	      if (auto *term = entry->getTerminator())
	        term->erase();
	      b.setInsertionPointToEnd(entry);

	      Value thisArg = entry->getArgument(0);
	      Value itemArg = entry->getArgument(1);

	      auto getTypeFnTy = FunctionType::get(getContext(), {thisArg.getType()},
	                                           {thisArg.getType()});
	      auto getTypeFn = intoModuleOp.lookupSymbol<mlir::func::FuncOp>(
          "circt_sv_class_get_type");
      if (!getTypeFn || getTypeFn.getFunctionType() != getTypeFnTy) {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(intoModuleOp.getBody());
        getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
        getTypeFn = mlir::func::FuncOp::create(builder, loc,
                                               "circt_sv_class_get_type",
                                               getTypeFnTy);
        getTypeFn.setPrivate();
        symbolTable.insert(getTypeFn);
      }
      Value dynType =
          mlir::func::CallOp::create(b, loc, getTypeFn, {thisArg}).getResult(0);

      auto cmpTy = moore::IntType::get(
          getContext(), /*width=*/32,
          cast<moore::IntType>(dynType.getType()).getDomain());

	      // Collect all `write` methods on user classes (non-UVM) that match the
	      // expected lowered signature: (i32 this, i32 item) -> void.
	      SmallVector<std::pair<const slang::ast::ClassType *, int32_t>>
	          classIdSnapshot;
	      classIdSnapshot.reserve(classIds.size());
	      for (auto &entry : classIds)
	        classIdSnapshot.push_back({entry.first, entry.second});

	      SmallVector<std::pair<int32_t, mlir::func::FuncOp>> impls;
	      for (auto [cls, classId] : classIdSnapshot) {
	        if (!cls)
	          continue;
	        llvm::StringRef clsName(cls->name.data(), cls->name.size());
	        if (clsName.starts_with("uvm_"))
          continue;
        const slang::ast::SubroutineSymbol *impl = nullptr;
        for (auto &method : cls->membersOfType<slang::ast::SubroutineSymbol>()) {
          if (llvm::StringRef(method.name.data(), method.name.size()) ==
              "write") {
            impl = &method;
            break;
          }
        }
        if (!impl)
          continue;
        if (failed(convertFunction(*impl)))
          continue;
        auto *implLowering = declareFunction(*impl);
        if (!implLowering || !implLowering->op)
          continue;
        if (implLowering->op.getFunctionType() != fnTy)
          continue;
        impls.push_back({classId, implLowering->op});
      }
      llvm::sort(impls, [](const auto &a, const auto &b) {
        return a.first < b.first;
      });

      mlir::func::FuncOp getImplFn;
      if (analysisImpId != 0) {
        auto getImplFnTy = FunctionType::get(getContext(), {i32Ty}, {i32Ty});
        getImplFn = intoModuleOp.lookupSymbol<mlir::func::FuncOp>(
            "circt_uvm_analysis_imp_get_impl");
        if (!getImplFn || getImplFn.getFunctionType() != getImplFnTy) {
          OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToStart(intoModuleOp.getBody());
          getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
          getImplFn = mlir::func::FuncOp::create(builder, loc,
                                                 "circt_uvm_analysis_imp_get_impl",
                                                 getImplFnTy);
          getImplFn.setPrivate();
          symbolTable.insert(getImplFn);
        }
      }

	      // Build a non-recursive dispatcher:
	      // - If the callee is an analysis_imp, resolve its impl and dispatch based
	      //   on the impl type.
	      // - Otherwise, dispatch based on the callee type directly.

      Block *impBlock = nullptr;
      if (analysisImpId != 0)
        impBlock = &dispatchFn.getBody().emplaceBlock();

      SmallVector<Block *> checkBlocks;
      SmallVector<Block *> caseBlocks;
      checkBlocks.reserve(impls.size());
      caseBlocks.reserve(impls.size());
      for (size_t i = 0, e = impls.size(); i < e; ++i) {
        Block *check = &dispatchFn.getBody().emplaceBlock();
        check->addArgument(i32Ty, loc); // target
        check->addArgument(cmpTy, loc); // dynType
        checkBlocks.push_back(check);

        Block *body = &dispatchFn.getBody().emplaceBlock();
        body->addArgument(i32Ty, loc); // target
        caseBlocks.push_back(body);
      }

      Block *defaultBlock = &dispatchFn.getBody().emplaceBlock();
      b.setInsertionPointToStart(defaultBlock);
      mlir::func::ReturnOp::create(b, loc);

      auto branchToDispatch =
          [&](OpBuilder &bb, Block *dest, Value target, Value type) {
            if (!dest) {
              mlir::cf::BranchOp::create(bb, loc, defaultBlock);
              return;
            }
            mlir::cf::BranchOp::create(bb, loc, dest, ValueRange{target, type});
          };

	      b.setInsertionPointToEnd(entry);
	      // Jump into the rebuilt dispatcher CFG.
	      if (checkBlocks.empty()) {
	        mlir::cf::BranchOp::create(b, loc, defaultBlock);
	      } else if (analysisImpId != 0) {
	        Value analysisIdVal =
	            moore::ConstantOp::create(b, loc, cmpTy, analysisImpId,
	                                     /*isSigned=*/true);
	        Value eq = b.createOrFold<moore::EqOp>(loc, dynType, analysisIdVal);
	        eq = b.createOrFold<moore::BoolCastOp>(loc, eq);
	        Value cond = moore::ToBuiltinBoolOp::create(b, loc, eq);
	        mlir::cf::CondBranchOp::create(b, loc, cond, impBlock, ValueRange{},
	                                       checkBlocks.front(),
	                                       ValueRange{thisArg, dynType});
	      } else {
	        mlir::cf::BranchOp::create(b, loc, checkBlocks.front(),
	                                   ValueRange{thisArg, dynType});
	      }

	      if (analysisImpId != 0) {
	        b.setInsertionPointToStart(impBlock);
	        if (getImplFn) {
          Value impl =
              mlir::func::CallOp::create(b, loc, getImplFn, {thisArg}).getResult(0);
          Value implType =
              mlir::func::CallOp::create(b, loc, getTypeFn, {impl}).getResult(0);
          branchToDispatch(b, checkBlocks.front(), impl, implType);
        } else {
          mlir::cf::BranchOp::create(b, loc, defaultBlock);
        }
      }

      for (size_t i = 0, e = impls.size(); i < e; ++i) {
        int32_t classId = impls[i].first;
        mlir::func::FuncOp implFn = impls[i].second;
        Block *next = (i + 1 < e) ? checkBlocks[i + 1] : defaultBlock;

        b.setInsertionPointToStart(checkBlocks[i]);
        Value target = checkBlocks[i]->getArgument(0);
        Value targetType = checkBlocks[i]->getArgument(1);
        Value classIdVal =
            moore::ConstantOp::create(b, loc, cmpTy, classId, /*isSigned=*/true);
        Value eq = b.createOrFold<moore::EqOp>(loc, targetType, classIdVal);
        eq = b.createOrFold<moore::BoolCastOp>(loc, eq);
        Value cond = moore::ToBuiltinBoolOp::create(b, loc, eq);
        ValueRange nextArgs =
            (next == defaultBlock) ? ValueRange{} : ValueRange{target, targetType};
        mlir::cf::CondBranchOp::create(b, loc, cond, caseBlocks[i],
                                       ValueRange{target}, next, nextArgs);

        b.setInsertionPointToStart(caseBlocks[i]);
        Value caseTarget = caseBlocks[i]->getArgument(0);
        mlir::func::CallOp::create(b, loc, implFn, ValueRange{caseTarget, itemArg});
        mlir::func::ReturnOp::create(b, loc);
      }

      return success();
    };

    if (failed(rebuildAnalysisDispatch()))
      return failure();
  }
  return success();
}

InterfaceLowering *
Context::convertInterface(const slang::ast::InstanceBodySymbol *interface) {
  using slang::ast::ArgumentDirection;

  for (auto const &existingIface : interfaces) {
    if (!existingIface.getFirst())
      continue;
    if (interface->hasSameType(*existingIface.getFirst())) {
      interface = existingIface.first;
      break;
    }
  }

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
    auto signalType = getInterfaceSignalType(type);
    if (!signalType || signalType == type) {
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
      auto &ifaceBody = slang::ast::InstanceBodySymbol::fromDefinition(
          compilation, *ifacePort->interfaceDef, ifacePort->location,
          slang::ast::InstanceFlags::None,
          /*hierarchyOverrideNode=*/nullptr, /*configBlock=*/nullptr,
          /*configRule=*/nullptr);
      auto *ifaceLowering = convertInterface(&ifaceBody);
      if (!ifaceLowering)
        return {};
      Type ifaceType = ifaceLowering->type;
      auto portName = builder.getStringAttr(ifacePort->name);
      modulePorts.push_back(
          hw::ModulePort{portName, ifaceType, hw::ModulePort::Input});
      auto arg = block->addArgument(ifaceType, portLoc);
      lowering.interfacePorts.push_back(
          ModuleLowering::InterfacePortLowering{*ifacePort, portLoc, arg});
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
    if (failed(member.visit(ModuleVisitor(*this, loc)))) {
      auto diag = mlir::emitError(loc) << "failed to convert module `"
                                       << module->name << "` member";
      if (!member.name.empty())
        diag << " `" << member.name << "`";
      diag << " (" << slang::ast::toString(member.kind) << ")";
      return failure();
    }
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

  // M3 bring-up: the UVM package is class-heavy and not yet fully supported.
  // However, some top-level entry points (e.g. `run_test`) are plain package
  // tasks/functions that we can lower and execute in top-executed mode. To
  // avoid being blocked by the full UVM surface area, selectively lower only
  // the small subset of package-level subroutines we currently rely on.
  if (package.name == "uvm_pkg") {
    ValueSymbolScope scope(valueSymbols);
    for (auto &member : package.members()) {
      if (member.kind != slang::ast::SymbolKind::Subroutine)
        continue;
      if (member.name != "run_test")
        continue;
      if (failed(convertFunction(member.as<slang::ast::SubroutineSymbol>())))
        return failure();
    }
    return success();
  }

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

int32_t Context::getOrAssignClassId(const slang::ast::ClassType &type) {
  auto it = classIds.find(&type);
  if (it != classIds.end())
    return it->second;
  int32_t id = nextClassId++;
  classIds[&type] = id;
  return id;
}

int32_t Context::getOrAssignClassFieldId(const slang::ast::VariableSymbol &field) {
  auto it = classFieldIds.find(&field);
  if (it != classFieldIds.end())
    return it->second;
  int32_t id = nextClassFieldId++;
  classFieldIds[&field] = id;
  return id;
}

int32_t Context::getOrAssignConstraintBlockId(
    const slang::ast::ConstraintBlockSymbol &block) {
  auto it = constraintBlockIds.find(&block);
  if (it != constraintBlockIds.end())
    return it->second;
  int32_t id = nextConstraintBlockId++;
  constraintBlockIds[&block] = id;
  return id;
}

/// Convert a function and its arguments to a function declaration in the IR.
/// This does not convert the function body.
FunctionLowering *
Context::declareFunction(const slang::ast::SubroutineSymbol &subroutine) {
  using slang::ast::ArgumentDirection;

  // Check if there already is a declaration for this function.
  auto &lowering = functions[&subroutine];
  if (lowering)
    return lowering.get();
  lowering = std::make_unique<FunctionLowering>();
  auto loc = convertLocation(subroutine.location);

  // Ensure the subroutine has been elaborated so members such as `thisVar` are
  // populated before we inspect them.
  auto astArgs = subroutine.getArguments();

  // Pick an insertion point for this function according to the source file
  // location.
  OpBuilder::InsertionGuard g(builder);
  auto it = orderedRootOps.upper_bound(subroutine.location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Bring-up mode: class methods (and UVM package subroutines) are treated as
  // absent and callers may stub them out.
  if (const auto *parentScope = subroutine.getParentScope()) {
    const auto &parentSym = parentScope->asSymbol();
    if (parentSym.kind == slang::ast::SymbolKind::ClassType ||
        parentSym.kind == slang::ast::SymbolKind::GenericClassDef) {
      if (options.allowClassStubs) {
        lowering->op = nullptr;
        return lowering.get();
      }
    }
    if (parentSym.kind == slang::ast::SymbolKind::Package &&
        parentSym.name == "uvm_pkg" && options.allowClassStubs) {
      lowering->op = nullptr;
      return lowering.get();
    }
  }
  if (subroutine.thisVar && options.allowClassStubs) {
    lowering->op = nullptr;
    return lowering.get();
  }

  // Determine the function type.
  SmallVector<Type> inputTypes;
  SmallVector<Type, 1> outputTypes;

  if (subroutine.thisVar) {
    auto type = convertType(subroutine.thisVar->getType());
    if (!type)
      return {};
    inputTypes.push_back(type);
  }

  for (const auto *arg : astArgs) {
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

  SmallString<64> funcName;
  if (const auto *parentScope = subroutine.getParentScope()) {
    const auto &parentSym = parentScope->asSymbol();
    if (parentSym.kind == slang::ast::SymbolKind::ClassType ||
        parentSym.kind == slang::ast::SymbolKind::GenericClassDef) {
      if (auto *grand = parentSym.getParentScope())
        guessNamespacePrefix(grand->asSymbol(), funcName);
      funcName += parentSym.name;
      funcName += "::";
      funcName += subroutine.name;
    } else {
      guessNamespacePrefix(parentSym, funcName);
      funcName += subroutine.name;
    }
  } else {
    funcName += subroutine.name;
  }

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
  if (subroutine.thisVar && options.allowClassStubs)
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
  if (!lowering->op.getBody().empty())
    return success();

  // DPI imports are external declarations; do not attempt to synthesize a body.
  if (subroutine.flags.has(slang::ast::MethodFlags::DPIImport))
    return success();
  ValueSymbolScope scope(valueSymbols);

  // Create a function body block and populate it with block arguments.
  auto astArgs = subroutine.getArguments();
  SmallVector<moore::VariableOp> argVariables;
  auto &block = lowering->op.getBody().emplaceBlock();
  auto inputTypes = lowering->op.getFunctionType().getInputs();
  size_t expectedInputs = astArgs.size() + (subroutine.thisVar ? 1 : 0);
  if (inputTypes.size() != expectedInputs) {
    mlir::emitError(lowering->op.getLoc(), "internal error: argument count mismatch for `")
        << lowering->op.getName() << "` (expected " << expectedInputs << ", got "
        << inputTypes.size() << ")";
    return failure();
  }
  size_t inputIdx = 0;
  bool pushedThis = false;
  auto thisGuard = llvm::make_scope_exit([&] {
    if (pushedThis) {
      thisStack.pop_back();
      thisClassStack.pop_back();
    }
  });
  if (subroutine.thisVar) {
    auto thisLoc = convertLocation(subroutine.thisVar->location);
    Value thisArg = block.addArgument(inputTypes[inputIdx++], thisLoc);
    valueSymbols.insert(subroutine.thisVar, thisArg);
    thisStack.push_back(thisArg);
    const slang::ast::ClassType *thisClass = nullptr;
    const slang::ast::Type &thisTy = subroutine.thisVar->getType();
    thisClass = thisTy.getCanonicalType().as_if<slang::ast::ClassType>();
    thisClassStack.push_back(thisClass);
    pushedThis = true;
  }

  for (const auto *astArg : astArgs) {
    auto loc = convertLocation(astArg->location);
    if (inputIdx >= inputTypes.size()) {
      mlir::emitError(lowering->op.getLoc(), "internal error: ran out of input types for `")
          << lowering->op.getName() << "`";
      return failure();
    }
    auto type = inputTypes[inputIdx++];
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

  functionReturnVarStack.push_back(returnVar);
  auto returnVarGuard =
      llvm::make_scope_exit([&] { functionReturnVarStack.pop_back(); });

  if (failed(convertStatement(subroutine.getBody())))
    return failure();

  // If there was no explicit return statement provided by the user, insert a
  // default one.
  if (builder.getBlock()) {
    auto resultTypes = lowering->op.getFunctionType().getResults();
    if (!resultTypes.empty()) {
      if (returnVar && !subroutine.getReturnType().isVoid()) {
        Value read =
            moore::ReadOp::create(builder, returnVar.getLoc(), returnVar);
        mlir::func::ReturnOp::create(builder, lowering->op.getLoc(), read);
      } else {
        // Some implicitly provided/built-in methods (e.g. class randomization
        // API) may not expose a `returnValVar`, but still have a non-void
        // signature. Materialize a conservative default return value so the IR
        // remains verifiable.
        Type resultType = resultTypes.front();
        Value defaultValue;
        if (auto intType = dyn_cast<moore::IntType>(resultType)) {
          int64_t value = 0;
          if (subroutine.name == "randomize")
            value = 1;
          defaultValue = moore::ConstantOp::create(builder, lowering->op.getLoc(),
                                                   intType, value,
                                                   /*isSigned=*/true);
        } else if (isa<moore::StringType>(resultType)) {
          auto i8Ty = moore::IntType::get(getContext(), /*width=*/8,
                                          moore::Domain::TwoValued);
          Value raw =
              moore::StringConstantOp::create(builder, lowering->op.getLoc(),
                                              i8Ty, "")
                  .getResult();
          defaultValue = moore::ConversionOp::create(builder, lowering->op.getLoc(),
                                                     resultType, raw);
        } else {
          mlir::emitError(lowering->op.getLoc(),
                          "unsupported default return type for `")
              << lowering->op.getName() << "`: " << resultType;
          return failure();
        }
        mlir::func::ReturnOp::create(builder, lowering->op.getLoc(),
                                     defaultValue);
      }
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
