//===- Statements.cpp - Slang statement conversion ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/SystemSubroutine.h"
#include "slang/ast/TimingControl.h"
#include "slang/ast/expressions/MiscExpressions.h"
#include "slang/ast/types/AllTypes.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace circt;
using namespace ImportVerilog;

// NOLINTBEGIN(misc-no-recursion)
namespace {
struct StmtVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  StmtVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  bool isTerminated() const { return !builder.getInsertionBlock(); }
  void setTerminated() { builder.clearInsertionPoint(); }

  Block &createBlock() {
    assert(builder.getInsertionBlock());
    auto block = std::make_unique<Block>();
    block->insertAfter(builder.getInsertionBlock());
    return *block.release();
  }

  LogicalResult recursiveForeach(const slang::ast::ForeachLoopStatement &stmt,
                                 uint32_t level) {
    // find current dimension we are operate.
    const auto &loopDim = stmt.loopDims[level];
    if (!loopDim.range.has_value())
      return mlir::emitError(loc) << "dynamic loop variable is unsupported";
    auto &exitBlock = createBlock();
    auto &stepBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&stepBlock, &exitBlock});
    llvm::scope_exit done([&] { context.loopStack.pop_back(); });

    const auto &iter = loopDim.loopVar;
    auto type = context.convertType(*iter->getDeclaredType());
    if (!type)
      return failure();

    Value initial = moore::ConstantOp::create(
        builder, loc, cast<moore::IntType>(type), loopDim.range->lower());

    // Create loop varirable in this dimension
    Value varOp = moore::VariableOp::create(
        builder, loc, moore::RefType::get(cast<moore::UnpackedType>(type)),
        builder.getStringAttr(iter->name), initial);
    context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                         iter, varOp);

    cf::BranchOp::create(builder, loc, &checkBlock);
    builder.setInsertionPointToEnd(&checkBlock);

    // When the loop variable is greater than the upper bound, goto exit
    auto upperBound = moore::ConstantOp::create(
        builder, loc, cast<moore::IntType>(type), loopDim.range->upper());

    auto var = moore::ReadOp::create(builder, loc, varOp);
    Value cond = moore::SleOp::create(builder, loc, var, upperBound);
    if (!cond)
      return failure();
    cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
    if (auto ty = dyn_cast<moore::IntType>(cond.getType());
        ty && ty.getDomain() == Domain::FourValued) {
      cond = moore::LogicToIntOp::create(builder, loc, cond);
    }
    cond = moore::ToBuiltinIntOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

    builder.setInsertionPointToEnd(&bodyBlock);

    // find next dimension in this foreach statement, it finded then recuersive
    // resolve, else perform body statement
    bool hasNext = false;
    for (uint32_t nextLevel = level + 1; nextLevel < stmt.loopDims.size();
         nextLevel++) {
      if (stmt.loopDims[nextLevel].loopVar) {
        if (failed(recursiveForeach(stmt, nextLevel)))
          return failure();
        hasNext = true;
        break;
      }
    }

    if (!hasNext) {
      if (failed(context.convertStatement(stmt.body)))
        return failure();
    }
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &stepBlock);

    builder.setInsertionPointToEnd(&stepBlock);

    // add one to loop variable
    var = moore::ReadOp::create(builder, loc, varOp);
    auto one =
        moore::ConstantOp::create(builder, loc, cast<moore::IntType>(type), 1);
    auto postValue = moore::AddOp::create(builder, loc, var, one).getResult();
    moore::BlockingAssignOp::create(builder, loc, varOp, postValue);
    cf::BranchOp::create(builder, loc, &checkBlock);

    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Skip empty statements (stray semicolons).
  LogicalResult visit(const slang::ast::EmptyStatement &) { return success(); }

  // Convert every statement in a statement list. The Verilog syntax follows a
  // similar philosophy as C/C++, where things like `if` and `for` accept a
  // single statement as body. But then a `{...}` block is a valid statement,
  // which allows for the `if {...}` syntax. In Verilog, things like `final`
  // accept a single body statement, but that can be a `begin ... end` block,
  // which in turn has a single body statement, which then commonly is a list of
  // statements.
  LogicalResult visit(const slang::ast::StatementList &stmts) {
    for (auto *stmt : stmts.list) {
      if (isTerminated()) {
        auto loc = context.convertLocation(stmt->sourceRange);
        mlir::emitWarning(loc, "unreachable code");
        break;
      }
      if (failed(context.convertStatement(*stmt))) {
        auto stmtLoc = context.convertLocation(stmt->sourceRange);
        mlir::emitError(stmtLoc)
            << "failed to convert statement "
            << slang::ast::toString(stmt->kind);
        return failure();
      }
    }
    return success();
  }

  // Inline `begin ... end` blocks into the parent.
  LogicalResult visit(const slang::ast::BlockStatement &stmt) {
    return context.convertStatement(stmt.body);
  }

  // Handle expression statements.
  LogicalResult visit(const slang::ast::ExpressionStatement &stmt) {
    // Special handling for calls to system tasks that return no result value.
    if (const auto *call = stmt.expr.as_if<slang::ast::CallExpression>()) {
      if (const auto *info =
              std::get_if<slang::ast::CallExpression::SystemCallInfo>(
                  &call->subroutine)) {
        auto handled = visitSystemCall(stmt, *call, *info);
        if (failed(handled))
          return failure();
        if (handled == true)
          return success();
      }

      // According to IEEE 1800-2023 Section 21.3.3 "Formatting data to a
      // string" the first argument of $sformat is its output; the other
      // arguments work like a FormatString.
      // In Moore we only support writing to a location if it is a reference;
      // However, Section 21.3.3 explains that the output of $sformat is
      // assigned as if it were cast from a string literal (Section 5.9),
      // so this implementation casts the string to the target value.
      if (!call->getSubroutineName().compare("$sformat")) {

        // Use the first argument as the output location
        auto *lhsExpr = call->arguments().front();
        // Format the second and all later arguments as a string
        auto fmtValue =
            context.convertFormatString(call->arguments().subspan(1), loc,
                                        moore::IntFormat::Decimal, false);
        if (failed(fmtValue))
          return failure();
        // Convert the FormatString to a StringType
        auto strValue = moore::FormatStringToStringOp::create(builder, loc,
                                                              fmtValue.value());
        // The Slang AST produces a `AssignmentExpression` for the first
        // argument; the RHS of this expression is invalid though
        // (`EmptyArgument`), so we only use the LHS of the
        // `AssignmentExpression` and plug in the formatted string for the RHS.
        if (auto assignExpr =
                lhsExpr->as_if<slang::ast::AssignmentExpression>()) {
          auto lhs = context.convertLvalueExpression(assignExpr->left());
          if (!lhs)
            return failure();

          auto convertedValue = context.materializeConversion(
              cast<moore::RefType>(lhs.getType()).getNestedType(), strValue,
              false, loc);
          moore::BlockingAssignOp::create(builder, loc, lhs, convertedValue);
          return success();
        } else {
          return failure();
        }
      }

      // Some system tasks may not be classified by slang as a SystemCallInfo
      // (and thus won't reach visitSystemCall). Handle `$timeformat` here as
      // a fallback.
      if (!call->getSubroutineName().compare("$timeformat")) {
        auto args = call->arguments();
        if (args.size() != 4)
          return emitError(loc) << "`$timeformat` expects 4 arguments";

        auto unitConst = context.evaluateConstant(*args[0]);
        auto precisionConst = context.evaluateConstant(*args[1]);
        auto suffixConst = context.evaluateConstant(*args[2]);
        auto minWidthConst = context.evaluateConstant(*args[3]);

        if (!unitConst.isInteger() || !precisionConst.isInteger() ||
            !minWidthConst.isInteger())
          return emitError(loc)
                 << "`$timeformat` unit/precision/minWidth must be integers";
        if (!suffixConst.isString())
          return emitError(loc) << "`$timeformat` suffix must be a string";

        auto unit = unitConst.integer().as<int32_t>();
        auto precision = precisionConst.integer().as<int32_t>();
        auto minWidth = minWidthConst.integer().as<int32_t>();
        if (!unit || !precision || !minWidth)
          return emitError(loc) << "`$timeformat` arguments out of range";

        auto suffixAttr = builder.getStringAttr(StringRef(suffixConst.str()));
        moore::TimeFormatBIOp::create(builder, loc, *unit, *precision,
                                      suffixAttr, *minWidth);
        return success();
      }
    }

    auto value = context.convertRvalueExpression(stmt.expr);
    if (!value) {
      auto diag = mlir::emitError(loc, "failed to convert expression statement");
      diag.attachNote(loc) << "expression kind: "
                           << slang::ast::toString(stmt.expr.kind);
      if (const auto *call = stmt.expr.as_if<slang::ast::CallExpression>()) {
        diag.attachNote(loc) << "call target: " << call->getSubroutineName();
        if (call->thisClass())
          diag.attachNote(loc) << "call has explicit receiver";
      }
      return failure();
    }

    // Expressions like calls to void functions return a dummy value that has no
    // uses. If the returned value is trivially dead, remove it.
    if (auto *defOp = value.getDefiningOp())
      if (isOpTriviallyDead(defOp))
        defOp->erase();

    return success();
  }

  // Handle variable declarations.
  LogicalResult visit(const slang::ast::VariableDeclStatement &stmt) {
    const auto &var = stmt.symbol;
    auto type = context.convertType(*var.getDeclaredType());
    if (!type)
      return failure();

    Value initial;
    if (const auto *init = var.getInitializer()) {
      initial = context.convertRvalueExpression(*init, type);
      if (!initial)
        return failure();
    } else {
      const slang::ast::Type &astType = var.getDeclaredType()->getType();

      auto getOrCreateExternFunc = [&](StringRef name, FunctionType fnType) {
        if (auto existing =
                context.intoModuleOp.lookupSymbol<mlir::func::FuncOp>(name)) {
          if (existing.getFunctionType() != fnType) {
            mlir::emitError(loc, "conflicting declarations for `")
                << name << "`";
            return mlir::func::FuncOp();
          }
          return existing;
        }

        OpBuilder::InsertionGuard g(context.builder);
        context.builder.setInsertionPointToStart(context.intoModuleOp.getBody());
        context.getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
        auto fn =
            mlir::func::FuncOp::create(context.builder, loc, name, fnType);
        fn.setPrivate();
        return fn;
      };

      // Runtime-backed dynamic containers default-initialize to an empty handle.
      if (astType.as_if<slang::ast::DynamicArrayType>()) {
        auto intType = dyn_cast<moore::IntType>(type);
        if (!intType) {
          mlir::emitError(loc, "unsupported dynamic array storage type: ")
              << type;
          return failure();
        }
        initial = moore::ConstantOp::create(builder, loc, intType, /*value=*/0,
                                            /*isSigned=*/true);
      } else if (astType.as_if<slang::ast::QueueType>()) {
        auto fnType = FunctionType::get(context.getContext(), {}, {type});
        auto fn = getOrCreateExternFunc("circt_sv_queue_alloc_i32", fnType);
        if (!fn)
          return failure();
        initial =
            mlir::func::CallOp::create(builder, loc, fn, ValueRange{})
                .getResult(0);
      } else if (astType.as_if<slang::ast::AssociativeArrayType>()) {
        auto fnType = FunctionType::get(context.getContext(), {}, {type});
        auto fn = getOrCreateExternFunc("circt_sv_assoc_alloc_str_i32", fnType);
        if (!fn)
          return failure();
        initial =
            mlir::func::CallOp::create(builder, loc, fn, ValueRange{})
                .getResult(0);
      }
    }

    // Collect local temporary variables.
    auto varOp = moore::VariableOp::create(
        builder, loc, moore::RefType::get(cast<moore::UnpackedType>(type)),
        builder.getStringAttr(var.name), initial);
    context.valueSymbols.insertIntoScope(context.valueSymbols.getCurScope(),
                                         &var, varOp);
    return success();
  }

  // Handle if statements.
  LogicalResult visit(const slang::ast::ConditionalStatement &stmt) {
    // Generate the condition. There may be multiple conditions linked with the
    // `&&&` operator.
    Value allConds;
    for (const auto &condition : stmt.conditions) {
      if (condition.pattern)
        return mlir::emitError(loc,
                               "match patterns in if conditions not supported");
      auto cond = context.convertRvalueExpression(*condition.expr);
      if (!cond)
        return failure();
      cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
      if (allConds)
        allConds = moore::AndOp::create(builder, loc, allConds, cond);
      else
        allConds = cond;
    }
    assert(allConds && "slang guarantees at least one condition");
    if (auto ty = dyn_cast<moore::IntType>(allConds.getType());
        ty && ty.getDomain() == Domain::FourValued) {
      allConds = moore::LogicToIntOp::create(builder, loc, allConds);
    }
    allConds = moore::ToBuiltinIntOp::create(builder, loc, allConds);

    // Create the blocks for the true and false branches, and the exit block.
    Block &exitBlock = createBlock();
    Block *falseBlock = stmt.ifFalse ? &createBlock() : nullptr;
    Block &trueBlock = createBlock();
    cf::CondBranchOp::create(builder, loc, allConds, &trueBlock,
                             falseBlock ? falseBlock : &exitBlock);

    // Generate the true branch.
    builder.setInsertionPointToEnd(&trueBlock);
    if (failed(context.convertStatement(stmt.ifTrue)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &exitBlock);

    // Generate the false branch if present.
    if (stmt.ifFalse) {
      builder.setInsertionPointToEnd(falseBlock);
      if (failed(context.convertStatement(*stmt.ifFalse)))
        return failure();
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);
    }

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  /// Handle case statements.
  LogicalResult visit(const slang::ast::CaseStatement &caseStmt) {
    using slang::ast::AttributeSymbol;
    using slang::ast::CaseStatementCondition;
    auto caseExpr = context.convertRvalueExpression(caseStmt.expr);
    if (!caseExpr)
      return failure();

    // Check each case individually. This currently ignores the `unique`,
    // `unique0`, and `priority` modifiers which would allow for additional
    // optimizations.
    auto &exitBlock = createBlock();
    Block *lastMatchBlock = nullptr;
    SmallVector<moore::FVIntegerAttr> itemConsts;

    for (const auto &item : caseStmt.items) {
      // Create the block that will contain the main body of the expression.
      // This is where any of the comparisons will branch to if they match.
      auto &matchBlock = createBlock();
      lastMatchBlock = &matchBlock;

      // The SV standard requires expressions to be checked in the order
      // specified by the user, and for the evaluation to stop as soon as the
      // first matching expression is encountered.
      for (const auto *expr : item.expressions) {
        auto value = context.convertRvalueExpression(*expr);
        if (!value)
          return failure();
        auto itemLoc = value.getLoc();

        // Take note if the expression is a constant.
        auto maybeConst = value;
        while (isa_and_nonnull<moore::ConversionOp, moore::IntToLogicOp,
                               moore::LogicToIntOp>(maybeConst.getDefiningOp()))
          maybeConst = maybeConst.getDefiningOp()->getOperand(0);
        if (auto defOp = maybeConst.getDefiningOp<moore::ConstantOp>())
          itemConsts.push_back(defOp.getValueAttr());

        // Generate the appropriate equality operator.
        Value cond;
        switch (caseStmt.condition) {
        case CaseStatementCondition::Normal:
          cond = moore::CaseEqOp::create(builder, itemLoc, caseExpr, value);
          break;
        case CaseStatementCondition::WildcardXOrZ:
          cond = moore::CaseXZEqOp::create(builder, itemLoc, caseExpr, value);
          break;
        case CaseStatementCondition::WildcardJustZ:
          cond = moore::CaseZEqOp::create(builder, itemLoc, caseExpr, value);
          break;
        case CaseStatementCondition::Inside:
          mlir::emitError(loc, "unsupported set membership case statement");
          return failure();
        }
        if (auto ty = dyn_cast<moore::IntType>(cond.getType());
            ty && ty.getDomain() == Domain::FourValued) {
          cond = moore::LogicToIntOp::create(builder, loc, cond);
        }
        cond = moore::ToBuiltinIntOp::create(builder, loc, cond);

        // If the condition matches, branch to the match block. Otherwise
        // continue checking the next expression in a new block.
        auto &nextBlock = createBlock();
        mlir::cf::CondBranchOp::create(builder, itemLoc, cond, &matchBlock,
                                       &nextBlock);
        builder.setInsertionPointToEnd(&nextBlock);
      }

      // The current block is the fall-through after all conditions have been
      // checked and nothing matched. Move the match block up before this point
      // to make the IR easier to read.
      matchBlock.moveBefore(builder.getInsertionBlock());

      // Generate the code for this item's statement in the match block.
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&matchBlock);
      if (failed(context.convertStatement(*item.stmt)))
        return failure();
      if (!isTerminated()) {
        auto loc = context.convertLocation(item.stmt->sourceRange);
        mlir::cf::BranchOp::create(builder, loc, &exitBlock);
      }
    }

    const auto caseStmtAttrs = context.compilation.getAttributes(caseStmt);
    const bool hasFullCaseAttr =
        llvm::find_if(caseStmtAttrs, [](const AttributeSymbol *attr) {
          return attr->name == "full_case";
        }) != caseStmtAttrs.end();

    // Check if the case statement looks exhaustive assuming two-state values.
    // We use this information to work around a common bug in input Verilog
    // where a case statement enumerates all possible two-state values of the
    // case expression, but forgets to deal with cases involving X and Z bits in
    // the input.
    //
    // Once the core dialects start supporting four-state values we may want to
    // tuck this behind an import option that is on by default, since it does
    // not preserve semantics.
    auto twoStateExhaustive = false;
    if (auto intType = dyn_cast<moore::IntType>(caseExpr.getType());
        intType && intType.getWidth() < 32 &&
        itemConsts.size() == (1 << intType.getWidth())) {
      // Sort the constants by value.
      llvm::sort(itemConsts, [](auto a, auto b) {
        return a.getValue().getRawValue().ult(b.getValue().getRawValue());
      });

      // Ensure that every possible value of the case expression is present. Do
      // this by starting at 0 and iterating over all sorted items. Each item
      // must be the previous item + 1. At the end, the addition must exactly
      // overflow and take us back to zero.
      auto nextValue = FVInt::getZero(intType.getWidth());
      for (auto value : itemConsts) {
        if (value.getValue() != nextValue)
          break;
        nextValue += 1;
      }
      twoStateExhaustive = nextValue.isZero();
    }

    // If the case statement is exhaustive assuming two-state values, don't
    // generate the default case. Instead, branch to the last match block. This
    // will essentially make the last case item the "default".
    //
    // Alternatively, if the case statement has an (* full_case *) attribute
    // but no default case, it indicates that the developer has intentionally
    // covered all known possible values. Hence, the last match block is
    // treated as the implicit "default" case.
    if ((twoStateExhaustive || (hasFullCaseAttr && !caseStmt.defaultCase)) &&
        lastMatchBlock &&
        caseStmt.condition == CaseStatementCondition::Normal) {
      mlir::cf::BranchOp::create(builder, loc, lastMatchBlock);
    } else {
      // Generate the default case if present.
      if (caseStmt.defaultCase)
        if (failed(context.convertStatement(*caseStmt.defaultCase)))
          return failure();
      if (!isTerminated())
        mlir::cf::BranchOp::create(builder, loc, &exitBlock);
    }

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  /// Handle `randcase` statements.
  ///
  /// These are used by the chapter-18 random stability tests. Lower them to a
  /// runtime selection based on `$urandom`, so that RNG state controls
  /// (`get_randstate` / `set_randstate`) can influence the choice.
  LogicalResult visit(const slang::ast::RandCaseStatement &stmt) {
    if (stmt.items.empty())
      return success();

    auto i32Ty =
        moore::IntType::get(context.getContext(), /*width=*/32, moore::Domain::TwoValued);

    SmallVector<Value> weights;
    weights.reserve(stmt.items.size());

    Value total =
        moore::ConstantOp::create(builder, loc, i32Ty, /*value=*/0, /*isSigned=*/true);
    for (const auto &item : stmt.items) {
      Value w = context.convertRvalueExpression(*item.expr);
      if (!w)
        return failure();
      w = context.materializeConversion(i32Ty, w, /*isSigned=*/false, loc);
      if (!w)
        return failure();
      weights.push_back(w);
      total = moore::AddOp::create(builder, loc, total, w).getResult();
    }

    Value zero =
        moore::ConstantOp::create(builder, loc, i32Ty, /*value=*/0, /*isSigned=*/true);
    Value nonZero = moore::NeOp::create(builder, loc, total, zero);
    nonZero = builder.createOrFold<moore::BoolCastOp>(loc, nonZero);
    nonZero = moore::ToBuiltinBoolOp::create(builder, loc, nonZero);

    Value r = moore::UrandomBIOp::create(builder, loc, nullptr);
    Value pick = moore::ModUOp::create(builder, loc, r, total).getResult();

    Block &exitBlock = createBlock();
    Block &checkBlock = createBlock();
    checkBlock.addArgument(i32Ty, loc);

    cf::CondBranchOp::create(builder, loc, nonZero, &checkBlock, ValueRange{pick},
                             &exitBlock, ValueRange{});

    builder.setInsertionPointToEnd(&checkBlock);
    Value remaining = checkBlock.getArgument(0);

    for (size_t i = 0, e = stmt.items.size(); i < e; ++i) {
      Block &matchBlock = createBlock();
      Block *nextBlock = nullptr;
      if (i + 1 < e) {
        nextBlock = &createBlock();
        nextBlock->addArgument(i32Ty, loc);
      } else {
        nextBlock = &exitBlock;
      }

      Value w = weights[i];
      Value cond = moore::UltOp::create(builder, loc, remaining, w);
      cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
      cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);

      Value nextRemaining = moore::SubOp::create(builder, loc, remaining, w).getResult();

      if (i + 1 < e)
        cf::CondBranchOp::create(builder, loc, cond, &matchBlock, ValueRange{},
                                 nextBlock, ValueRange{nextRemaining});
      else
        cf::CondBranchOp::create(builder, loc, cond, &matchBlock, nextBlock);

      // Generate the selected item.
      builder.setInsertionPointToEnd(&matchBlock);
      if (failed(context.convertStatement(*stmt.items[i].stmt)))
        return failure();
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);

      // Continue with the next item.
      if (i + 1 < e) {
        builder.setInsertionPointToEnd(nextBlock);
        remaining = nextBlock->getArgument(0);
      }
    }

    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Handle `for` loops.
  LogicalResult visit(const slang::ast::ForLoopStatement &stmt) {
    // Generate the initializers.
    for (auto *initExpr : stmt.initializers)
      if (!context.convertRvalueExpression(*initExpr))
        return failure();

    // Create the blocks for the loop condition, body, step, and exit.
    auto &exitBlock = createBlock();
    auto &stepBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();
    cf::BranchOp::create(builder, loc, &checkBlock);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&stepBlock, &exitBlock});
    llvm::scope_exit done([&] { context.loopStack.pop_back(); });

    // Generate the loop condition check.
    builder.setInsertionPointToEnd(&checkBlock);
    auto cond = context.convertRvalueExpression(*stmt.stopExpr);
    if (!cond)
      return failure();
    cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
    if (auto ty = dyn_cast<moore::IntType>(cond.getType());
        ty && ty.getDomain() == Domain::FourValued) {
      cond = moore::LogicToIntOp::create(builder, loc, cond);
    }
    cond = moore::ToBuiltinIntOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &stepBlock);

    // Generate the step expressions.
    builder.setInsertionPointToEnd(&stepBlock);
    for (auto *stepExpr : stmt.steps)
      if (!context.convertRvalueExpression(*stepExpr))
        return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &checkBlock);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  LogicalResult visit(const slang::ast::ForeachLoopStatement &stmt) {
    for (uint32_t level = 0; level < stmt.loopDims.size(); level++) {
      if (stmt.loopDims[level].loopVar)
        return recursiveForeach(stmt, level);
    }
    return success();
  }

  // Handle `repeat` loops.
  LogicalResult visit(const slang::ast::RepeatLoopStatement &stmt) {
    auto count = context.convertRvalueExpression(stmt.count);
    if (!count)
      return failure();

    // Create the blocks for the loop condition, body, step, and exit.
    auto &exitBlock = createBlock();
    auto &stepBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();
    auto currentCount = checkBlock.addArgument(count.getType(), count.getLoc());
    cf::BranchOp::create(builder, loc, &checkBlock, count);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&stepBlock, &exitBlock});
    llvm::scope_exit done([&] { context.loopStack.pop_back(); });

    // Generate the loop condition check.
    builder.setInsertionPointToEnd(&checkBlock);
    auto cond = builder.createOrFold<moore::BoolCastOp>(loc, currentCount);
    if (auto ty = dyn_cast<moore::IntType>(cond.getType());
        ty && ty.getDomain() == Domain::FourValued) {
      cond = moore::LogicToIntOp::create(builder, loc, cond);
    }
    cond = moore::ToBuiltinIntOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &stepBlock);

    // Decrement the current count and branch back to the check block.
    builder.setInsertionPointToEnd(&stepBlock);
    auto one = moore::ConstantOp::create(
        builder, count.getLoc(), cast<moore::IntType>(count.getType()), 1);
    Value nextCount =
        moore::SubOp::create(builder, count.getLoc(), currentCount, one);
    cf::BranchOp::create(builder, loc, &checkBlock, nextCount);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Handle `while` and `do-while` loops.
  LogicalResult createWhileLoop(const slang::ast::Expression &condExpr,
                                const slang::ast::Statement &bodyStmt,
                                bool atLeastOnce) {
    // Create the blocks for the loop condition, body, and exit.
    auto &exitBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &checkBlock = createBlock();
    cf::BranchOp::create(builder, loc, atLeastOnce ? &bodyBlock : &checkBlock);
    if (atLeastOnce)
      bodyBlock.moveBefore(&checkBlock);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&checkBlock, &exitBlock});
    llvm::scope_exit done([&] { context.loopStack.pop_back(); });

    // Generate the loop condition check.
    builder.setInsertionPointToEnd(&checkBlock);
    auto cond = context.convertRvalueExpression(condExpr);
    if (!cond)
      return failure();
    cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
    if (auto ty = dyn_cast<moore::IntType>(cond.getType());
        ty && ty.getDomain() == Domain::FourValued) {
      cond = moore::LogicToIntOp::create(builder, loc, cond);
    }
    cond = moore::ToBuiltinIntOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &exitBlock);

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(bodyStmt)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &checkBlock);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  LogicalResult visit(const slang::ast::WhileLoopStatement &stmt) {
    return createWhileLoop(stmt.cond, stmt.body, false);
  }

  LogicalResult visit(const slang::ast::DoWhileLoopStatement &stmt) {
    return createWhileLoop(stmt.cond, stmt.body, true);
  }

  // Handle `forever` loops.
  LogicalResult visit(const slang::ast::ForeverLoopStatement &stmt) {
    // Create the blocks for the loop body and exit.
    auto &exitBlock = createBlock();
    auto &bodyBlock = createBlock();
    cf::BranchOp::create(builder, loc, &bodyBlock);

    // Push the blocks onto the loop stack such that we can continue and break.
    context.loopStack.push_back({&bodyBlock, &exitBlock});
    llvm::scope_exit done([&] { context.loopStack.pop_back(); });

    // Generate the loop body.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(stmt.body)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &bodyBlock);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Handle timing control.
  LogicalResult visit(const slang::ast::TimedStatement &stmt) {
    return context.convertTimingControl(stmt.timing, stmt.stmt);
  }

  // Handle `wait(<expr>) <stmt>;`.
  LogicalResult visit(const slang::ast::WaitStatement &stmt) {
    // Create the blocks for the wait condition check, the suspend point, the
    // post-wait statement, and the exit.
    auto &exitBlock = createBlock();
    auto &bodyBlock = createBlock();
    auto &waitBlock = createBlock();
    auto &checkBlock = createBlock();
    cf::BranchOp::create(builder, loc, &checkBlock);

    // Evaluate the condition in the check block, collecting all variables read
    // as part of the expression so we can observe them for changes while
    // waiting.
    builder.setInsertionPointToEnd(&checkBlock);
    llvm::SmallSetVector<Value, 8> observedValues;
    Value cond;
    {
      auto previousCallback = context.rvalueReadCallback;
      auto done = llvm::make_scope_exit(
          [&] { context.rvalueReadCallback = previousCallback; });
      context.rvalueReadCallback = [&](moore::ReadOp readOp) {
        observedValues.insert(readOp.getInput());
        if (previousCallback)
          previousCallback(readOp);
      };
      cond = context.convertRvalueExpression(stmt.cond);
      if (!cond)
        return failure();
    }
    cond = builder.createOrFold<moore::BoolCastOp>(loc, cond);
    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);
    cf::CondBranchOp::create(builder, loc, cond, &bodyBlock, &waitBlock);

    // Suspend in the wait block until any of the observed values changes, then
    // re-check the condition.
    builder.setInsertionPointToEnd(&waitBlock);
    auto waitOp = moore::WaitEventOp::create(builder, loc);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&waitOp.getBody().emplaceBlock());
      auto previousCallback = context.rvalueReadCallback;
      auto done = llvm::make_scope_exit(
          [&] { context.rvalueReadCallback = previousCallback; });
      // Reads performed solely to populate the wait op should not be reported
      // to any surrounding implicit event collection.
      context.rvalueReadCallback = nullptr;
      for (auto observed : observedValues) {
        auto value = moore::ReadOp::create(builder, loc, observed);
        moore::DetectEventOp::create(builder, loc, moore::Edge::AnyChange,
                                     value, Value{});
      }
    }
    cf::BranchOp::create(builder, loc, &checkBlock);

    // Convert the trailing statement once the wait has completed.
    builder.setInsertionPointToEnd(&bodyBlock);
    if (failed(context.convertStatement(stmt.stmt)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &exitBlock);

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);
    }
    return success();
  }

  // Handle `->e;` and `->>e;` event triggers.
  //
  // We currently model SystemVerilog `event` values as an integer token. A
  // trigger increments the token, and `@(e)` is lowered as an any-change event
  // control on that token.
  LogicalResult visit(const slang::ast::EventTriggerStatement &stmt) {
    if (stmt.timing)
      return mlir::emitError(loc) << "unsupported timed event trigger";

    // Class property events are lowered through the runtime-backed class object
    // model, and thus do not have a direct `ref` storage location. Handle them
    // here by reading the current token from the class object and writing back
    // the incremented value.
    if (auto *named = stmt.target.as_if<slang::ast::NamedValueExpression>()) {
      if (auto *prop =
              named->symbol.as_if<slang::ast::ClassPropertySymbol>()) {
        auto fieldType = context.convertType(prop->getType());
        if (!fieldType)
          return failure();
        auto intTy = dyn_cast<moore::IntType>(fieldType);
        if (!intTy || !intTy.getBitSize().has_value() ||
            *intTy.getBitSize() != 32) {
          auto d = mlir::emitError(loc, "unsupported class event trigger type: ")
                   << fieldType;
          d.attachNote(context.convertLocation(prop->location))
              << "property declared here";
          return failure();
        }

        auto getOrCreateExternFunc = [&](StringRef name, FunctionType fnType) {
          if (auto existing =
                  context.intoModuleOp.lookupSymbol<mlir::func::FuncOp>(name)) {
            if (existing.getFunctionType() != fnType) {
              mlir::emitError(loc, "conflicting declarations for `")
                  << name << "`";
              return mlir::func::FuncOp();
            }
            return existing;
          }
          OpBuilder::InsertionGuard g(context.builder);
          context.builder.setInsertionPointToStart(context.intoModuleOp.getBody());
          context.getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
          auto fn =
              mlir::func::FuncOp::create(context.builder, loc, name, fnType);
          fn.setPrivate();
          return fn;
        };

        auto i32Ty = moore::IntType::get(context.getContext(), /*width=*/32,
                                         moore::Domain::TwoValued);
        Value handleVal;
        if (prop->lifetime == slang::ast::VariableLifetime::Static) {
          handleVal = moore::ConstantOp::create(builder, loc, i32Ty, /*value=*/0,
                                                /*isSigned=*/true);
        } else {
          if (context.thisStack.empty()) {
            auto d = mlir::emitError(loc, "event trigger on class property `")
                     << prop->name << "` without a `this` handle";
            d.attachNote(context.convertLocation(prop->location))
                << "property declared here";
            return failure();
          }
          handleVal = context.thisStack.back();
          handleVal = context.materializeConversion(i32Ty, handleVal,
                                                    /*isSigned=*/false, loc);
          if (!handleVal)
            return failure();
        }

        int32_t fieldId = context.getOrAssignClassFieldId(*prop);
        Value fieldIdVal =
            moore::ConstantOp::create(builder, loc, i32Ty, fieldId,
                                      /*isSigned=*/true);

        auto getFnType =
            FunctionType::get(context.getContext(), {i32Ty, i32Ty}, {i32Ty});
        auto getFn = getOrCreateExternFunc("circt_sv_class_get_i32", getFnType);
        if (!getFn)
          return failure();
        Value cur = mlir::func::CallOp::create(builder, loc, getFn,
                                               {handleVal, fieldIdVal})
                        .getResult(0);

        Value one = moore::ConstantOp::create(builder, loc, i32Ty, 1,
                                              /*isSigned=*/true);
        Value next = moore::AddOp::create(builder, loc, cur, one).getResult();

        auto setFnType = FunctionType::get(context.getContext(),
                                           {i32Ty, i32Ty, i32Ty}, {});
        auto setFn =
            getOrCreateExternFunc("circt_sv_class_set_i32", setFnType);
        if (!setFn)
          return failure();
        builder.create<mlir::func::CallOp>(loc, setFn,
                                           ValueRange{handleVal, fieldIdVal, next});
        return success();
      }
    }

    Value lhs = context.convertLvalueExpression(stmt.target);
    if (!lhs)
      return failure();
    auto refTy = dyn_cast<moore::RefType>(lhs.getType());
    if (!refTy)
      return mlir::emitError(loc) << "event trigger target did not lower to a ref";

    auto intTy = dyn_cast<moore::IntType>(refTy.getNestedType());
    if (!intTy)
      return mlir::emitError(loc) << "unsupported event trigger target type";

    Value cur = moore::ReadOp::create(builder, loc, lhs);
    Value one = moore::ConstantOp::create(builder, loc, intTy, 1);
    Value next = moore::AddOp::create(builder, loc, cur, one).getResult();
    moore::BlockingAssignOp::create(builder, loc, lhs, next);
    return success();
  }

  // Handle return statements.
  LogicalResult visit(const slang::ast::ReturnStatement &stmt) {
    Operation *parentOp =
        builder.getInsertionBlock() ? builder.getInsertionBlock()->getParentOp()
                                    : nullptr;
    if (!parentOp)
      return mlir::emitError(loc) << "return statement is not within an op";

    // Return from an MLIR `func.func` body.
    if (auto funcOp = dyn_cast<mlir::func::FuncOp>(parentOp)) {
      auto resultTypes = funcOp.getFunctionType().getResults();
      if (resultTypes.size() > 1)
        return mlir::emitError(loc)
               << "unsupported function return arity: " << resultTypes.size();

      if (stmt.expr) {
        if (resultTypes.empty())
          return mlir::emitError(loc)
                 << "cannot return a value from a void function";
        auto expr = context.convertRvalueExpression(*stmt.expr, resultTypes[0]);
        if (!expr)
          return failure();
        mlir::func::ReturnOp::create(builder, loc, ValueRange{expr});
      } else {
        if (resultTypes.empty()) {
          mlir::func::ReturnOp::create(builder, loc, ValueRange{});
        } else {
          if (context.functionReturnVarStack.empty() ||
              !context.functionReturnVarStack.back())
            return mlir::emitError(loc)
                   << "missing return variable for non-void function";
          Value read = moore::ReadOp::create(
              builder, loc, context.functionReturnVarStack.back());
          mlir::func::ReturnOp::create(builder, loc, ValueRange{read});
        }
      }
      setTerminated();
      return success();
    }

    // Return from an SV dialect `sv.func` body.
    if (isa<sv::FuncOp>(parentOp)) {
      if (stmt.expr) {
        auto expr = context.convertRvalueExpression(*stmt.expr);
        if (!expr)
          return failure();
        sv::ReturnOp::create(builder, loc, ValueRange{expr});
      } else {
        sv::ReturnOp::create(builder, loc, ValueRange{});
      }
      setTerminated();
      return success();
    }

    // Return from module-level procedures (initial/always/tasks).
    if (isa<moore::ProcedureOp>(parentOp)) {
      if (stmt.expr)
        return mlir::emitError(loc)
               << "unsupported `return <expr>` in a procedure";
      moore::ReturnOp::create(builder, loc);
      setTerminated();
      return success();
    }

    return mlir::emitError(loc) << "unsupported return statement context";
  }

  // Handle continue statements.
  LogicalResult visit(const slang::ast::ContinueStatement &stmt) {
    if (context.loopStack.empty())
      return mlir::emitError(loc,
                             "cannot `continue` without a surrounding loop");
    cf::BranchOp::create(builder, loc, context.loopStack.back().continueBlock);
    setTerminated();
    return success();
  }

  // Handle break statements.
  LogicalResult visit(const slang::ast::BreakStatement &stmt) {
    if (context.loopStack.empty())
      return mlir::emitError(loc, "cannot `break` without a surrounding loop");
    cf::BranchOp::create(builder, loc, context.loopStack.back().breakBlock);
    setTerminated();
    return success();
  }

  // Handle immediate assertion statements.
  LogicalResult visit(const slang::ast::ImmediateAssertionStatement &stmt) {
    // Handle assertion statements that don't have an action block.
    if (stmt.ifTrue && stmt.ifTrue->as_if<slang::ast::EmptyStatement>()) {
      auto cond = context.convertRvalueExpression(stmt.cond);
      cond = context.convertToBool(cond);
      if (!cond)
        return failure();

      // Moore assertion ops currently require a `moore.procedure` parent. When
      // an immediate assertion appears inside a function/task body that is
      // lowered as an isolated region (e.g. `func.func`), keep the condition
      // evaluation for side effects but otherwise drop the assertion.
      if (!builder.getInsertionBlock() ||
          !isa<moore::ProcedureOp>(builder.getInsertionBlock()->getParentOp()))
        return success();

      auto defer = moore::DeferAssert::Immediate;
      if (stmt.isFinal)
        defer = moore::DeferAssert::Final;
      else if (stmt.isDeferred)
        defer = moore::DeferAssert::Observed;

      switch (stmt.assertionKind) {
      case slang::ast::AssertionKind::Assert:
        moore::AssertOp::create(builder, loc, defer, cond, StringAttr{});
        return success();
      case slang::ast::AssertionKind::Assume:
        moore::AssumeOp::create(builder, loc, defer, cond, StringAttr{});
        return success();
      case slang::ast::AssertionKind::CoverProperty:
        moore::CoverOp::create(builder, loc, defer, cond, StringAttr{});
        return success();
      default:
        break;
      }
      mlir::emitError(loc) << "unsupported immediate assertion kind: "
                           << slang::ast::toString(stmt.assertionKind);
      return failure();
    }

    // Regard assertion statements with an action block as the "if-else".
    moore::ProcedureOp procOp;
    if (auto *block = builder.getInsertionBlock())
      procOp = dyn_cast<moore::ProcedureOp>(block->getParentOp());

    // Some SV frontends desugar concurrent assertions with action blocks into a
    // clocked procedural block containing an immediate assertion statement.
    // Model sampled-value functions (`$past`, `$rose`, ...) in this procedural
    // form by synthesizing per-assertion state.
    DenseMap<const slang::ast::Expression *, Value> prevVars;
    DenseMap<const slang::ast::Expression *, Value> curValues;
    unsigned nextPrevId = 0;
    Value havePastVar;
    auto scopeId = context.nextAssertionCallScopeId++;
    auto prefix = ("__svtests_sa" + Twine(scopeId)).str();

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
          casted = moore::LogicToIntOp::create(builder, value.getLoc(), casted);
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
      if (!procOp)
        return {};

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&procOp.getBody().front());
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
      if (!argExpr || !procOp)
        return {};
      if (Value existing = prevVars.lookup(argExpr))
        return existing;

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&procOp.getBody().front());
      Value init0 = moore::ConstantOp::create(builder, declLoc, intTy, 0);
      auto name = builder.getStringAttr(
          (Twine(prefix) + "_prev" + Twine(nextPrevId++)).str());
      Value var = moore::VariableOp::create(
          builder, declLoc,
          moore::RefType::get(cast<moore::UnpackedType>(intTy)), name, init0);
      prevVars.insert({argExpr, var});
      return var;
    };

    Value cond;
    {
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

      cond = context.convertRvalueExpression(stmt.cond);
      cond = context.convertToBool(cond);
      if (!cond)
        return failure();
    }

    cond = moore::ToBuiltinBoolOp::create(builder, loc, cond);

    // Create the blocks for the true and false branches, and the exit block.
    Block &exitBlock = createBlock();
    Block *falseBlock = stmt.ifFalse ? &createBlock() : nullptr;
    Block &trueBlock = createBlock();
    cf::CondBranchOp::create(builder, loc, cond, &trueBlock,
                             falseBlock ? falseBlock : &exitBlock);

    // Generate the true branch.
    builder.setInsertionPointToEnd(&trueBlock);
    if (stmt.ifTrue && failed(context.convertStatement(*stmt.ifTrue)))
      return failure();
    if (!isTerminated())
      cf::BranchOp::create(builder, loc, &exitBlock);

    if (stmt.ifFalse) {
      // Generate the false branch if present.
      builder.setInsertionPointToEnd(falseBlock);
      if (failed(context.convertStatement(*stmt.ifFalse)))
        return failure();
      if (!isTerminated())
        cf::BranchOp::create(builder, loc, &exitBlock);
    }

    // If control never reaches the exit block, remove it and mark control flow
    // as terminated. Otherwise we continue inserting ops in the exit block.
    if (exitBlock.hasNoPredecessors()) {
      exitBlock.erase();
      setTerminated();
    } else {
      builder.setInsertionPointToEnd(&exitBlock);

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
    }
    return success();
  }

  // Handle concurrent assertion statements.
  LogicalResult visit(const slang::ast::ConcurrentAssertionStatement &stmt) {
    auto loc = context.convertLocation(stmt.sourceRange);

    // Check for a `disable iff` expression:
    // The DisableIff construct can only occcur at the top level of an assertion
    // and cannot be nested within properties.
    // Hence we only need to detect if the top level assertion expression
    // has type DisableIff, negate the `disable` expression, then pass it to
    // the `enable` parameter of AssertOp/AssumeOp.
    Value enable;
    const slang::ast::AssertionExpr *propertySpec = &stmt.propertySpec;
    if (auto *disableIff =
            stmt.propertySpec.as_if<slang::ast::DisableIffAssertionExpr>()) {
      Value disableCond = context.convertRvalueExpression(disableIff->condition);
      disableCond = context.convertToBool(disableCond, Domain::TwoValued);
      if (!disableCond)
        return failure();

      Value enableCond = moore::NotOp::create(builder, loc, disableCond);
      enable = context.convertToI1(enableCond);
      propertySpec = &disableIff->expr;
    }
    auto isEmptyAction = [](const slang::ast::Statement *s) -> bool {
      return !s || s->as_if<slang::ast::EmptyStatement>();
    };

    // Best-effort extraction of a clock and boolean condition from a concurrent
    // assertion property spec, including named sequences/properties (which are
    // represented as assertion-instance expressions).
    auto extractClockedCondition =
        [&](const slang::ast::AssertionExpr &root,
            const slang::ast::SignalEventControl *&clockCtrl,
            const slang::ast::Expression *&condExpr) -> void {
      clockCtrl = nullptr;
      condExpr = nullptr;

      const slang::ast::AssertionExpr *expr = &root;
      // Keep this conservative; we only want simple, single-cycle sampling.
      for (unsigned depth = 0; depth < 64 && expr; ++depth) {
        if (auto *clocked = expr->as_if<slang::ast::ClockingAssertionExpr>()) {
          if (clocked->clocking.kind !=
              slang::ast::TimingControlKind::SignalEvent)
            break;
          clockCtrl = &clocked->clocking.as<slang::ast::SignalEventControl>();
          expr = &clocked->expr;
          continue;
        }

        if (auto *withMatch =
                expr->as_if<slang::ast::SequenceWithMatchExpr>()) {
          if (withMatch->repetition.has_value() ||
              !withMatch->matchItems.empty())
            break;
          expr = &withMatch->expr;
          continue;
        }

        if (auto *concat = expr->as_if<slang::ast::SequenceConcatExpr>()) {
          if (concat->elements.size() != 1)
            break;
          const auto &elem = concat->elements.front();
          if (elem.delay.min != 1 || (elem.delay.max && *elem.delay.max != 1))
            break;
          expr = elem.sequence;
          continue;
        }

        if (auto *strongWeak =
                expr->as_if<slang::ast::StrongWeakAssertionExpr>()) {
          expr = &strongWeak->expr;
          continue;
        }

        if (auto *simple = expr->as_if<slang::ast::SimpleAssertionExpr>()) {
          if (simple->repetition.has_value())
            break;

          // If we already found a sampling clock, this is the boolean condition.
          if (clockCtrl) {
            condExpr = &simple->expr;
            break;
          }

          // Otherwise, try to unwrap named sequences/properties:
          //   assert property (seq) else ...
          // where `seq` binds to an AssertionInstanceExpression whose body
          // contains the clocking event and condition.
          if (auto *inst =
                  simple->expr.as_if<slang::ast::AssertionInstanceExpression>()) {
            if (!inst->arguments.empty())
              break;
            expr = &inst->body;
            continue;
          }

          break;
        }

        break;
      }
    };

    // If the concurrent assertion has action blocks, best-effort lower it to a
    // clocked procedural check so side effects (e.g. `uvm_info` in `else`)
    // appear in simulation.
    bool hasTrueAction = !isEmptyAction(stmt.ifTrue);
    bool hasFalseAction = !isEmptyAction(stmt.ifFalse);
    if (hasTrueAction || hasFalseAction) {
      struct FixedSeqTerm {
        const slang::ast::Expression *expr = nullptr;
        uint32_t time = 0;
      };
      struct FixedSeqInfo {
        uint32_t length = 0;
        SmallVector<FixedSeqTerm, 8> terms;
      };

      // Best-effort extraction of a sampling clock and the underlying
      // assertion body, including named sequences/properties.
      auto extractClockedBody =
          [&](const slang::ast::AssertionExpr &root,
              const slang::ast::SignalEventControl *&clockCtrl,
              const slang::ast::AssertionExpr *&body) -> void {
        clockCtrl = nullptr;
        body = nullptr;

        const slang::ast::AssertionExpr *expr = &root;
        for (unsigned depth = 0; depth < 64 && expr; ++depth) {
          if (auto *clocked =
                  expr->as_if<slang::ast::ClockingAssertionExpr>()) {
            if (clocked->clocking.kind !=
                slang::ast::TimingControlKind::SignalEvent)
              break;
            clockCtrl = &clocked->clocking.as<slang::ast::SignalEventControl>();
            expr = &clocked->expr;
            continue;
          }

          if (auto *withMatch =
                  expr->as_if<slang::ast::SequenceWithMatchExpr>()) {
            if (withMatch->repetition.has_value() ||
                !withMatch->matchItems.empty())
              break;
            expr = &withMatch->expr;
            continue;
          }

          if (auto *strongWeak =
                  expr->as_if<slang::ast::StrongWeakAssertionExpr>()) {
            expr = &strongWeak->expr;
            continue;
          }

          if (auto *simple = expr->as_if<slang::ast::SimpleAssertionExpr>()) {
            if (simple->repetition.has_value())
              break;
            if (auto *inst = simple->expr.as_if<
                    slang::ast::AssertionInstanceExpression>()) {
              if (!inst->arguments.empty())
                break;
              expr = &inst->body;
              continue;
            }
            break;
          }

          break;
        }

        if (clockCtrl)
          body = expr;
      };

      // Best-effort analysis of a fixed-delay, match-item-free sequence into a
      // list of boolean terms with absolute cycle offsets and a fixed length.
      //
      // This is intentionally conservative; it exists to allow action blocks
      // on simple sequences (e.g. `assert property (a ##5 b) else ...`) to
      // execute under simulation even when declarative LTL lowering is not
      // supported.
      auto analyzeFixedSequence =
          [&](const slang::ast::AssertionExpr &root) -> std::optional<FixedSeqInfo> {
        std::function<std::optional<FixedSeqInfo>(const slang::ast::AssertionExpr &)>
            analyze = [&](const slang::ast::AssertionExpr &expr)
            -> std::optional<FixedSeqInfo> {
          if (auto *strongWeak =
                  expr.as_if<slang::ast::StrongWeakAssertionExpr>())
            return analyze(strongWeak->expr);

          if (auto *withMatch = expr.as_if<slang::ast::SequenceWithMatchExpr>()) {
            if (withMatch->repetition.has_value() || !withMatch->matchItems.empty())
              return std::nullopt;
            return analyze(withMatch->expr);
          }

          if (auto *concat = expr.as_if<slang::ast::SequenceConcatExpr>()) {
            FixedSeqInfo out;
            uint32_t cursor = 0;
            for (const auto &elem : concat->elements) {
              if (elem.delay.max && *elem.delay.max != elem.delay.min)
                return std::nullopt;
              uint32_t delay = elem.delay.min;
              if (!elem.sequence)
                return std::nullopt;
              auto child = analyze(*elem.sequence);
              if (!child)
                return std::nullopt;
              cursor += delay;
              for (auto term : child->terms)
                out.terms.push_back({term.expr, cursor + term.time});
              cursor += child->length;
            }
            out.length = cursor;
            return out;
          }

          if (auto *binary = expr.as_if<slang::ast::BinaryAssertionExpr>()) {
            using slang::ast::BinaryAssertionOperator;
            if (binary->op != BinaryAssertionOperator::And)
              return std::nullopt;
            auto lhs = analyze(binary->left);
            auto rhs = analyze(binary->right);
            if (!lhs || !rhs)
              return std::nullopt;
            FixedSeqInfo out;
            out.length = std::max(lhs->length, rhs->length);
            out.terms.append(lhs->terms.begin(), lhs->terms.end());
            out.terms.append(rhs->terms.begin(), rhs->terms.end());
            return out;
          }

          if (auto *simple = expr.as_if<slang::ast::SimpleAssertionExpr>()) {
            if (simple->repetition.has_value())
              return std::nullopt;
            if (auto *inst = simple->expr.as_if<
                    slang::ast::AssertionInstanceExpression>()) {
              if (!inst->arguments.empty())
                return std::nullopt;
              return analyze(inst->body);
            }
            FixedSeqInfo out;
            out.length = 0;
            out.terms.push_back({&simple->expr, 0});
            return out;
          }

          return std::nullopt;
        };

        return analyze(root);
      };

      auto savedIP = builder.saveInsertionPoint();
      bool restoreInsertionPoint = false;
      auto restoreIP = llvm::make_scope_exit([&] {
        if (restoreInsertionPoint)
          builder.restoreInsertionPoint(savedIP);
      });

      moore::ProcedureOp procOp;
      if (auto *block = builder.getInsertionBlock())
        procOp = dyn_cast<moore::ProcedureOp>(block->getParentOp());
      moore::SVModuleOp svModule;
      if (auto *block = builder.getInsertionBlock()) {
        if (auto *parentOp = block->getParentOp()) {
          if (auto direct = dyn_cast<moore::SVModuleOp>(parentOp))
            svModule = direct;
          else
            svModule = parentOp->getParentOfType<moore::SVModuleOp>();
        }
      }
      const slang::ast::SignalEventControl *clockCtrl = nullptr;
      const slang::ast::Expression *condExpr = nullptr;
      extractClockedCondition(*propertySpec, clockCtrl, condExpr);
      const slang::ast::AssertionExpr *seqBody = nullptr;
      std::optional<FixedSeqInfo> fixedSeq;
      if (!condExpr) {
        const slang::ast::SignalEventControl *seqClockCtrl = nullptr;
        extractClockedBody(*propertySpec, seqClockCtrl, seqBody);
        if (seqClockCtrl && seqBody) {
          fixedSeq = analyzeFixedSequence(*seqBody);
          if (fixedSeq)
            clockCtrl = seqClockCtrl;
        }
      }

      // Module-scope concurrent assertions are not inherently procedural.
      // Create a dedicated `always` block so action blocks (e.g. `else
      // uvm_info(...)`) can execute in simulation.
      if (!procOp && svModule && clockCtrl && (condExpr || fixedSeq)) {
        restoreInsertionPoint = true;
        Block &modBody = svModule.getBodyRegion().front();
        if (auto *term = modBody.getTerminator())
          builder.setInsertionPoint(term);
        else
          builder.setInsertionPointToEnd(&modBody);
        procOp = moore::ProcedureOp::create(builder, loc,
                                            moore::ProcedureKind::Always);
        builder.setInsertionPointToEnd(&procOp.getBody().emplaceBlock());
      }

      if (procOp && clockCtrl && condExpr) {
        Context::ValueSymbolScope scope(context.valueSymbols);

        unsigned scopeId = context.nextAssertionCallScopeId++;
        auto prefix = ("__svtests_ca" + Twine(scopeId)).str();

        // Create a per-assertion "has past sample" flag.
        Value havePastVar;
        {
          OpBuilder::InsertionGuard guard(builder);
          // Store sampled-value state at module scope so it persists across
          // activations of the generated `always` process.
          Block &modBody = svModule.getBodyRegion().front();
          builder.setInsertionPointToStart(&modBody);
          auto havePastTy =
              moore::IntType::get(context.getContext(), 1, Domain::TwoValued);
          Value init0 = moore::ConstantOp::create(builder, loc, havePastTy, 0);
          havePastVar = moore::VariableOp::create(
              builder, loc,
              moore::RefType::get(cast<moore::UnpackedType>(havePastTy)),
              builder.getStringAttr(prefix + "_has_past"), init0);
        }

        DenseMap<const slang::ast::Expression *, Value> prevVars;
        DenseMap<const slang::ast::Expression *, Value> curValues;
        unsigned nextPrevId = 0;

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
            casted =
                moore::ReplicateOp::create(builder, value.getLoc(), targetType,
                                           casted);
          }
          return casted;
        };

        // Override lowering of assertion-only system calls within the
        // assertion condition to avoid emitting LTL, and instead model them in
        // terms of local past-sample state.
        auto savedOverride = context.assertionCallOverride;
        context.assertionCallOverride =
            [&](const slang::ast::CallExpression &expr,
                const slang::ast::CallExpression::SystemCallInfo &info,
                Location callLoc) -> Value {
          const auto &subroutine = *info.subroutine;
          auto args = expr.arguments();
          if (args.size() != 1) {
            mlir::emitError(callLoc, "unsupported assertion call `")
                << subroutine.name << "` arity: expected 1, got " << args.size();
            return {};
          }
          if (!args[0]) {
            mlir::emitError(callLoc, "unsupported assertion call `")
                << subroutine.name << "`: null argument";
            return {};
          }

          // Convert the argument expression under the same override. Nested
          // assertion calls are handled recursively.
          Value argVal = context.convertRvalueExpression(*args[0]);
          if (!argVal)
            return {};

          auto intTy = dyn_cast<moore::IntType>(argVal.getType());
          if (!intTy) {
            mlir::emitError(callLoc, "unsupported assertion call `")
                << subroutine.name << "` operand type: " << argVal.getType();
            return {};
          }

          const slang::ast::Expression *argExpr = args[0];
          Value prevVar = prevVars.lookup(argExpr);
          if (!prevVar) {
            OpBuilder::InsertionGuard guard(builder);
            Block &modBody = svModule.getBodyRegion().front();
            builder.setInsertionPointToStart(&modBody);
            Value init0 = moore::ConstantOp::create(builder, callLoc, intTy, 0);
            auto name = builder.getStringAttr(
                (Twine(prefix) + "_prev" + Twine(nextPrevId++)).str());
            prevVar = moore::VariableOp::create(
                builder, callLoc,
                moore::RefType::get(cast<moore::UnpackedType>(intTy)), name,
                init0);
            prevVars.insert({argExpr, prevVar});
          }

          // Record the current value for updating past state at the end of the
          // sampling tick.
          if (!curValues.count(argExpr))
            curValues.insert({argExpr, argVal});

          Value havePast = moore::ReadOp::create(builder, callLoc, havePastVar);
          Value prevVal = moore::ReadOp::create(builder, callLoc, prevVar);

          // Helper to AND a single-bit guard with an arbitrary-width value by
          // replicating and matching domain.
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
            // Approximation: `$past(x)` is `x[-1]` if a past sample exists,
            // otherwise 0.
            return guardAnd(havePast, prevVal);
          }

          if (subroutine.name == "$stable") {
            // `$stable(x)` is defined only once a past sample exists.
            Value eq = moore::EqOp::create(builder, callLoc, argVal, prevVal);
            auto eqTy = cast<moore::IntType>(eq.getType());
            Value havePastBit = toMatchingIntType(havePast, eqTy);
            if (!havePastBit)
              return {};
            return moore::AndOp::create(builder, callLoc, havePastBit, eq);
          }

          if (subroutine.name == "$changed") {
            // `$changed(x)` is the complement of `$stable(x)`.
            Value eq = moore::EqOp::create(builder, callLoc, argVal, prevVal);
            auto eqTy = cast<moore::IntType>(eq.getType());
            Value havePastBit = toMatchingIntType(havePast, eqTy);
            if (!havePastBit)
              return {};
            Value stable =
                moore::AndOp::create(builder, callLoc, havePastBit, eq);
            return moore::NotOp::create(builder, callLoc, stable);
          }

          // $rose/$fell are defined on single-bit arguments.
          auto width = intTy.getBitSize();
          if (!width || *width != 1) {
            mlir::emitError(callLoc, "unsupported assertion call `")
                << subroutine.name
                << "` on non-1-bit operand of type: " << intTy;
            return {};
          }

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

          mlir::emitError(callLoc) << "unsupported assertion call `"
                                   << subroutine.name << "`";
          return {};
        };
        auto restoreOverride = llvm::make_scope_exit(
            [&] { context.assertionCallOverride = savedOverride; });

        // Sample the assertion condition *before* waiting for the clock edge.
        //
        // SVA sampled-value semantics observe the values "just before" the
        // clocking event (preponed). We do not model simulator regions, so we
        // approximate this by sampling between edges. For signals that change
        // only on the sampling edge, the between-edge value matches the
        // pre-edge sampled value.
        //
        // Convert the assertion condition with the assertion call override in
        // place so sampled-value functions (`$past`, `$rose`, ...) use the same
        // pre-edge snapshot.
        Value condMoore = context.convertRvalueExpression(*condExpr);
        if (!condMoore)
          return failure();
        condMoore = context.convertToBool(condMoore);
        if (!condMoore)
          return failure();

        // Suspend until the sampling clock edge occurs.
        auto waitOp = moore::WaitEventOp::create(builder, loc);
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&waitOp.getBody().emplaceBlock());

          auto toEdge = [](slang::ast::EdgeKind edge) -> moore::Edge {
            using slang::ast::EdgeKind;
            switch (edge) {
            case EdgeKind::None:
              return moore::Edge::AnyChange;
            case EdgeKind::PosEdge:
              return moore::Edge::PosEdge;
            case EdgeKind::NegEdge:
              return moore::Edge::NegEdge;
            case EdgeKind::BothEdges:
              return moore::Edge::BothEdges;
            }
            llvm_unreachable("all edge kinds handled");
          };

          Value clkExpr = context.convertRvalueExpression(clockCtrl->expr);
          if (!clkExpr)
            return failure();
          Value iffCond;
          if (clockCtrl->iffCondition) {
            iffCond = context.convertRvalueExpression(*clockCtrl->iffCondition);
            iffCond = context.convertToBool(iffCond, Domain::TwoValued);
            if (!iffCond)
              return failure();
          }
          moore::DetectEventOp::create(builder, loc, toEdge(clockCtrl->edge),
                                       clkExpr, iffCond);
        }

        Value cond = moore::ToBuiltinBoolOp::create(builder, loc, condMoore);

        // Create the blocks for the true and false branches, and the merge
        // block for state update.
        Block &mergeBlock = createBlock();
        Block *falseBlock = hasFalseAction ? &createBlock() : nullptr;
        Block &trueBlock = createBlock();
        cf::CondBranchOp::create(builder, loc, cond, &trueBlock,
                                 falseBlock ? falseBlock : &mergeBlock);

        builder.setInsertionPointToEnd(&trueBlock);
        if (hasTrueAction) {
          if (failed(context.convertStatement(*stmt.ifTrue)))
            return failure();
        }
        if (!isTerminated())
          cf::BranchOp::create(builder, loc, &mergeBlock);

        if (falseBlock) {
          builder.setInsertionPointToEnd(falseBlock);
          if (failed(context.convertStatement(*stmt.ifFalse)))
            return failure();
          if (!isTerminated())
            cf::BranchOp::create(builder, loc, &mergeBlock);
        }

        if (mergeBlock.hasNoPredecessors()) {
          mergeBlock.erase();
          setTerminated();
          return success();
        }

        builder.setInsertionPointToEnd(&mergeBlock);

        // Update stored past values at the end of the sampling tick.
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

        moore::ReturnOp::create(builder, loc);
        setTerminated();
        return success();
      }

      // Lower `expect (seq) ...` statements with action blocks in a procedural
      // context by unrolling a fixed-length, fixed-delay sequence. Unlike
      // `assert property`, `expect` is blocking and only runs once.
      if (procOp && svModule && clockCtrl && fixedSeq &&
          stmt.assertionKind == slang::ast::AssertionKind::Expect &&
          fixedSeq->length <= 64 && !fixedSeq->terms.empty()) {
        Context::ValueSymbolScope scope(context.valueSymbols);

        auto toEdge = [](slang::ast::EdgeKind edge) -> moore::Edge {
          using slang::ast::EdgeKind;
          switch (edge) {
          case EdgeKind::None:
            return moore::Edge::AnyChange;
          case EdgeKind::PosEdge:
            return moore::Edge::PosEdge;
          case EdgeKind::NegEdge:
            return moore::Edge::NegEdge;
          case EdgeKind::BothEdges:
            return moore::Edge::BothEdges;
          }
          llvm_unreachable("all edge kinds handled");
        };

        auto emitWaitForClock = [&]() -> LogicalResult {
          auto waitOp = moore::WaitEventOp::create(builder, loc);
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&waitOp.getBody().emplaceBlock());

          Value clkExpr = context.convertRvalueExpression(clockCtrl->expr);
          if (!clkExpr)
            return failure();
          Value iffCond;
          if (clockCtrl->iffCondition) {
            iffCond = context.convertRvalueExpression(*clockCtrl->iffCondition);
            iffCond = context.convertToBool(iffCond, Domain::TwoValued);
            if (!iffCond)
              return failure();
          }
          moore::DetectEventOp::create(builder, loc, toEdge(clockCtrl->edge),
                                       clkExpr, iffCond);
          return success();
        };

        auto seqLen = fixedSeq->length;
        auto bitTy =
            moore::IntType::get(context.getContext(), 1, Domain::TwoValued);
        Value init1 = moore::ConstantOp::create(builder, loc, bitTy, 1);

        // Bucket the fixed-sequence terms by absolute cycle offset.
        SmallVector<SmallVector<const slang::ast::Expression *, 4>> termsAt;
        termsAt.resize(seqLen + 1);
        for (auto term : fixedSeq->terms) {
          if (!term.expr)
            return failure();
          if (term.time > seqLen)
            continue;
          termsAt[term.time].push_back(term.expr);
        }

        Block &mergeBlock = createBlock();
        for (uint32_t step = 0; step <= seqLen; ++step) {
          // Sample the step's terms before waiting for the clock edge. For
          // signals that change only on the clock edge, this yields the
          // pre-edge values expected by SVA sampled-value semantics.
          Value stepCondMoore = init1;
          for (auto *expr : termsAt[step]) {
            Value cur = context.convertRvalueExpression(*expr);
            if (!cur)
              return failure();
            cur = context.convertToBool(cur, Domain::TwoValued);
            if (!cur)
              return failure();
            stepCondMoore = moore::AndOp::create(builder, loc, stepCondMoore, cur);
          }

          if (failed(emitWaitForClock()))
            return failure();

          Value stepCond = moore::ToBuiltinBoolOp::create(builder, loc, stepCondMoore);
          Block &okBlock = createBlock();
          Block &failBlock = createBlock();
          cf::CondBranchOp::create(builder, loc, stepCond, &okBlock, &failBlock);

          builder.setInsertionPointToEnd(&failBlock);
          if (hasFalseAction) {
            if (failed(context.convertStatement(*stmt.ifFalse)))
              return failure();
          }
          if (!isTerminated())
            cf::BranchOp::create(builder, loc, &mergeBlock);

          builder.setInsertionPointToEnd(&okBlock);
          if (step == seqLen) {
            if (hasTrueAction) {
              if (failed(context.convertStatement(*stmt.ifTrue)))
                return failure();
            }
            if (!isTerminated())
              cf::BranchOp::create(builder, loc, &mergeBlock);
            break;
          }
        }

        if (mergeBlock.hasNoPredecessors()) {
          mergeBlock.erase();
          setTerminated();
          return success();
        }

        builder.setInsertionPointToEnd(&mergeBlock);
        return success();
      }

      if (procOp && svModule && clockCtrl && fixedSeq &&
          fixedSeq->length <= 64 && !fixedSeq->terms.empty()) {
        Context::ValueSymbolScope scope(context.valueSymbols);

        auto seqLen = fixedSeq->length;

        unsigned scopeId = context.nextAssertionCallScopeId++;
        auto prefix = ("__svtests_ca" + Twine(scopeId)).str();

        // Collect unique term expressions for history tracking.
        SmallVector<const slang::ast::Expression *> uniqExprs;
        for (auto term : fixedSeq->terms) {
          if (!term.expr)
            return failure();
          if (llvm::find(uniqExprs, term.expr) == uniqExprs.end())
            uniqExprs.push_back(term.expr);
        }

        auto bitTy =
            moore::IntType::get(context.getContext(), 1, Domain::TwoValued);
        Value init1 = moore::ConstantOp::create(builder, loc, bitTy, 1);

        SmallVector<Value> validVars;
        DenseMap<const slang::ast::Expression *, SmallVector<Value>> histVars;
        {
          OpBuilder::InsertionGuard guard(builder);
          Block &modBody = svModule.getBodyRegion().front();
          builder.setInsertionPointToStart(&modBody);
          Value init0 = moore::ConstantOp::create(builder, loc, bitTy, 0);

          // History-valid shift register (tracks when we have `seqLen` past samples).
          validVars.reserve(seqLen + 1);
          for (uint32_t i = 0; i <= seqLen; ++i) {
            auto name = builder.getStringAttr((Twine(prefix) + "_valid" + Twine(i)).str());
            validVars.push_back(moore::VariableOp::create(
                builder, loc, moore::RefType::get(cast<moore::UnpackedType>(bitTy)),
                name, init0));
          }

          // Per-term history shift registers (store sampled values as booleans).
          unsigned exprIdx = 0;
          for (auto *expr : uniqExprs) {
            SmallVector<Value> vars;
            vars.reserve(seqLen + 1);
            for (uint32_t i = 0; i <= seqLen; ++i) {
              auto name = builder.getStringAttr(
                  (Twine(prefix) + "_h" + Twine(exprIdx) + "_" + Twine(i)).str());
              vars.push_back(moore::VariableOp::create(
                  builder, loc,
                  moore::RefType::get(cast<moore::UnpackedType>(bitTy)), name,
                  init0));
            }
            histVars.insert({expr, std::move(vars)});
            ++exprIdx;
          }
        }

        // Sample all tracked expression values *before* the clock edge. For
        // designs that update state using nonblocking assignments on the same
        // edge, this avoids sampling post-update values when modeling
        // sampled-value semantics in a procedural form.
        DenseMap<const slang::ast::Expression *, Value> preSamples;
        preSamples.reserve(uniqExprs.size());
        for (auto *expr : uniqExprs) {
          Value cur = context.convertRvalueExpression(*expr);
          if (!cur)
            return failure();
          cur = context.convertToBool(cur, Domain::TwoValued);
          if (!cur)
            return failure();
          preSamples.insert({expr, cur});
        }

        // Suspend until the sampling clock edge occurs.
        auto waitOp = moore::WaitEventOp::create(builder, loc);
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&waitOp.getBody().emplaceBlock());

          auto toEdge = [](slang::ast::EdgeKind edge) -> moore::Edge {
            using slang::ast::EdgeKind;
            switch (edge) {
            case EdgeKind::None:
              return moore::Edge::AnyChange;
            case EdgeKind::PosEdge:
              return moore::Edge::PosEdge;
            case EdgeKind::NegEdge:
              return moore::Edge::NegEdge;
            case EdgeKind::BothEdges:
              return moore::Edge::BothEdges;
            }
            llvm_unreachable("all edge kinds handled");
          };

          Value clkExpr = context.convertRvalueExpression(clockCtrl->expr);
          if (!clkExpr)
            return failure();
          Value iffCond;
          if (clockCtrl->iffCondition) {
            iffCond = context.convertRvalueExpression(*clockCtrl->iffCondition);
            iffCond = context.convertToBool(iffCond, Domain::TwoValued);
            if (!iffCond)
              return failure();
          }
          moore::DetectEventOp::create(builder, loc, toEdge(clockCtrl->edge),
                                       clkExpr, iffCond);
        }

        // Shift the "valid history" marker.
        for (uint32_t i = seqLen; i > 0; --i) {
          Value prev = moore::ReadOp::create(builder, loc, validVars[i - 1]);
          moore::BlockingAssignOp::create(builder, loc, validVars[i], prev);
        }
        moore::BlockingAssignOp::create(builder, loc, validVars[0], init1);
        Value haveHistory = moore::ReadOp::create(builder, loc, validVars[seqLen]);

        // Sample and shift all tracked expression histories.
        for (auto *expr : uniqExprs) {
          auto it = histVars.find(expr);
          if (it == histVars.end())
            continue;
          auto &vars = it->second;
          Value cur = preSamples.lookup(expr);
          if (!cur)
            return failure();

          for (uint32_t i = seqLen; i > 0; --i) {
            Value prev = moore::ReadOp::create(builder, loc, vars[i - 1]);
            moore::BlockingAssignOp::create(builder, loc, vars[i], prev);
          }
          moore::BlockingAssignOp::create(builder, loc, vars[0], cur);
        }

        // Compute match condition at the end of the fixed-length sequence.
        Value matchCond = init1;
        for (auto term : fixedSeq->terms) {
          auto it = histVars.find(term.expr);
          if (it == histVars.end())
            continue;
          uint32_t offset = seqLen >= term.time ? (seqLen - term.time) : 0;
          Value v = moore::ReadOp::create(builder, loc, it->second[offset]);
          matchCond = moore::AndOp::create(builder, loc, matchCond, v);
        }

        // If we have insufficient history, treat the assertion as satisfied to
        // avoid spurious failures at time 0.
        Value notHaveHistory = moore::NotOp::create(builder, loc, haveHistory);
        Value condMoore = moore::OrOp::create(builder, loc, notHaveHistory, matchCond);
        Value cond = moore::ToBuiltinBoolOp::create(builder, loc, condMoore);

        Block &mergeBlock = createBlock();
        Block *falseBlock = hasFalseAction ? &createBlock() : nullptr;
        Block &trueBlock = createBlock();
        cf::CondBranchOp::create(builder, loc, cond, &trueBlock,
                                 falseBlock ? falseBlock : &mergeBlock);

        builder.setInsertionPointToEnd(&trueBlock);
        if (hasTrueAction) {
          if (failed(context.convertStatement(*stmt.ifTrue)))
            return failure();
        }
        if (!isTerminated())
          cf::BranchOp::create(builder, loc, &mergeBlock);

        if (falseBlock) {
          builder.setInsertionPointToEnd(falseBlock);
          if (failed(context.convertStatement(*stmt.ifFalse)))
            return failure();
          if (!isTerminated())
            cf::BranchOp::create(builder, loc, &mergeBlock);
        }

        if (mergeBlock.hasNoPredecessors()) {
          mergeBlock.erase();
          setTerminated();
          return success();
        }

        builder.setInsertionPointToEnd(&mergeBlock);
        moore::ReturnOp::create(builder, loc);
        setTerminated();
        return success();
      }
    }

    // Fallback: preserve the declarative assertion for verification flows.
    auto property = context.convertAssertionExpression(*propertySpec, loc);
    if (!property)
      return failure();

    switch (stmt.assertionKind) {
    case slang::ast::AssertionKind::Assert:
    case slang::ast::AssertionKind::Expect:
      verif::AssertOp::create(builder, loc, property, enable, StringAttr{});
      return success();
    case slang::ast::AssertionKind::Assume:
    case slang::ast::AssertionKind::Restrict:
      verif::AssumeOp::create(builder, loc, property, enable, StringAttr{});
      return success();
    case slang::ast::AssertionKind::CoverProperty:
    case slang::ast::AssertionKind::CoverSequence:
      verif::CoverOp::create(builder, loc, property, enable, StringAttr{});
      return success();
    }
    mlir::emitError(loc) << "unsupported concurrent assertion kind: "
                         << slang::ast::toString(stmt.assertionKind);
    return failure();
  }

  // According to 1800-2023 Section 21.2.1 "The display and write tasks":
  // >> The $display and $write tasks display their arguments in the same
  // >> order as they appear in the argument list. Each argument can be a
  // >> string literal or an expression that returns a value.
  // According to Section 20.10 "Severity system tasks", the same
  // semantics apply to $fatal, $error, $warning, and $info.
  // This means we must first check whether the first "string-able"
  // argument is a Literal Expression which doesn't represent a fully-formatted
  // string, otherwise we convert it to a FormatStringType.
  FailureOr<Value>
  getDisplayMessage(std::span<const slang::ast::Expression *const> args) {
    if (args.size() == 0)
      return Value{};

    // Handle the string formatting.
    // If the second argument is a Literal of some type, we should either
    // treat it as a literal-to-be-formatted or a FormatStringType.
    // In this check we use a StringLiteral, but slang allows casting between
    // any literal expressions (strings, integers, reals, and time at least) so
    // this is short-hand for "any value literal"
    if (args[0]->as_if<slang::ast::StringLiteral>()) {
      return context.convertFormatString(args, loc);
    }
    // Check if there's only one argument and it's a FormatStringType
    if (args.size() == 1) {
      return context.convertRvalueExpression(
          *args[0], builder.getType<moore::FormatStringType>());
    }
    // Otherwise this looks invalid. Raise an error.
    return emitError(loc) << "Failed to convert Display Message!";
  }

  /// Handle the subset of system calls that return no result value. Return
  /// true if the called system task could be handled, false otherwise. Return
  /// failure if an error occurred.
  FailureOr<bool>
  visitSystemCall(const slang::ast::ExpressionStatement &stmt,
                  const slang::ast::CallExpression &expr,
                  const slang::ast::CallExpression::SystemCallInfo &info) {
    const auto &subroutine = *info.subroutine;
    auto args = expr.arguments();

    // Simulation Control Tasks

    if (subroutine.name == "$stop") {
      createFinishMessage(args.size() >= 1 ? args[0] : nullptr);
      moore::StopBIOp::create(builder, loc);
      return true;
    }

    if (subroutine.name == "$finish") {
      createFinishMessage(args.size() >= 1 ? args[0] : nullptr);
      moore::FinishBIOp::create(builder, loc, 0);
      moore::UnreachableOp::create(builder, loc);
      setTerminated();
      return true;
    }

    if (subroutine.name == "$exit") {
      // Calls to `$exit` from outside a `program` are ignored. Since we don't
      // yet support programs, there is nothing to do here.
      // TODO: Fix this once we support programs.
      return true;
    }

    if (subroutine.name == "$timeformat") {
      if (args.size() != 4)
        return emitError(loc) << "`$timeformat` expects 4 arguments";

      auto unitConst = context.evaluateConstant(*args[0]);
      auto precisionConst = context.evaluateConstant(*args[1]);
      auto suffixConst = context.evaluateConstant(*args[2]);
      auto minWidthConst = context.evaluateConstant(*args[3]);

      if (!unitConst.isInteger() || !precisionConst.isInteger() ||
          !minWidthConst.isInteger())
        return emitError(loc)
               << "`$timeformat` unit/precision/minWidth must be integers";
      if (!suffixConst.isString())
        return emitError(loc) << "`$timeformat` suffix must be a string";

      auto unit = unitConst.integer().as<int32_t>();
      auto precision = precisionConst.integer().as<int32_t>();
      auto minWidth = minWidthConst.integer().as<int32_t>();
      if (!unit || !precision || !minWidth)
        return emitError(loc) << "`$timeformat` arguments out of range";

      auto suffixAttr =
          builder.getStringAttr(StringRef(suffixConst.str()));
      moore::TimeFormatBIOp::create(builder, loc, *unit, *precision, suffixAttr,
                                    *minWidth);
      return true;
    }

    // Display and Write Tasks (`$display[boh]?` or `$write[boh]?`)

    // Check for a `$display` or `$write` prefix.
    bool isDisplay = false;     // display or write
    bool appendNewline = false; // display
    StringRef remainingName = subroutine.name;
    if (remainingName.consume_front("$display")) {
      isDisplay = true;
      appendNewline = true;
    } else if (remainingName.consume_front("$write")) {
      isDisplay = true;
    }

    // Check for optional `b`, `o`, or `h` suffix indicating default format.
    using moore::IntFormat;
    IntFormat defaultFormat = IntFormat::Decimal;
    if (isDisplay && !remainingName.empty()) {
      if (remainingName == "b")
        defaultFormat = IntFormat::Binary;
      else if (remainingName == "o")
        defaultFormat = IntFormat::Octal;
      else if (remainingName == "h")
        defaultFormat = IntFormat::HexLower;
      else
        isDisplay = false;
    }

    if (isDisplay) {
      auto message =
          context.convertFormatString(args, loc, defaultFormat, appendNewline);
      if (failed(message))
        return failure();
      if (*message == Value{})
        return true;
      moore::DisplayBIOp::create(builder, loc, *message);
      return true;
    }

    // Severity Tasks
    using moore::Severity;
    std::optional<Severity> severity;
    if (subroutine.name == "$info")
      severity = Severity::Info;
    else if (subroutine.name == "$warning")
      severity = Severity::Warning;
    else if (subroutine.name == "$error")
      severity = Severity::Error;
    else if (subroutine.name == "$fatal")
      severity = Severity::Fatal;

    if (severity) {
      // The `$fatal` task has an optional leading verbosity argument.
      const slang::ast::Expression *verbosityExpr = nullptr;
      if (severity == Severity::Fatal && args.size() >= 1) {
        verbosityExpr = args[0];
        args = args.subspan(1);
      }

      FailureOr<Value> maybeMessage = getDisplayMessage(args);
      if (failed(maybeMessage))
        return failure();
      auto message = maybeMessage.value();

      if (message == Value{})
        message = moore::FormatLiteralOp::create(builder, loc, "");
      moore::SeverityBIOp::create(builder, loc, *severity, message);

      // Handle the `$fatal` case which behaves like a `$finish`.
      if (severity == Severity::Fatal) {
        createFinishMessage(verbosityExpr);
        moore::FinishBIOp::create(builder, loc, 1);
        moore::UnreachableOp::create(builder, loc);
        setTerminated();
      }
      return true;
    }

    // Queue Tasks

    if (args.size() >= 1 && args[0]->type->isQueue()) {
      auto queue = context.convertLvalueExpression(*args[0]);

      // `delete` has two functions: If there is an index passed, then it
      // deletes that specific element, otherwise, it clears the entire queue.
      if (subroutine.name == "delete") {
        if (args.size() == 1) {
          moore::QueueClearOp::create(builder, loc, queue);
          return true;
        }
        if (args.size() == 2) {
          auto index = context.convertRvalueExpression(*args[1]);
          moore::QueueDeleteOp::create(builder, loc, queue, index);
          return true;
        }
      } else if (subroutine.name == "insert" && args.size() == 3) {
        auto index = context.convertRvalueExpression(*args[1]);
        auto item = context.convertRvalueExpression(*args[2]);

        moore::QueueInsertOp::create(builder, loc, queue, index, item);
        return true;
      } else if (subroutine.name == "push_back" && args.size() == 2) {
        auto item = context.convertRvalueExpression(*args[1]);
        moore::QueuePushBackOp::create(builder, loc, queue, item);
        return true;
      } else if (subroutine.name == "push_front" && args.size() == 2) {
        auto item = context.convertRvalueExpression(*args[1]);
        moore::QueuePushFrontOp::create(builder, loc, queue, item);
        return true;
      }

      return false;
    }

    // Give up on any other system tasks. These will be tried again as an
    // expression later.
    return false;
  }

  /// Create the optional diagnostic message print for finish-like ops.
  void createFinishMessage(const slang::ast::Expression *verbosityExpr) {
    unsigned verbosity = 1;
    if (verbosityExpr) {
      auto value =
          context.evaluateConstant(*verbosityExpr).integer().as<unsigned>();
      assert(value && "Slang guarantees constant verbosity parameter");
      verbosity = *value;
    }
    if (verbosity == 0)
      return;
    moore::FinishMessageBIOp::create(builder, loc, verbosity > 1);
  }

  // Handle event trigger statements.
  LogicalResult visit(const slang::ast::EventTriggerStatement &stmt) {
    if (stmt.timing) {
      mlir::emitError(loc) << "unsupported delayed event trigger";
      return failure();
    }

    // Events are lowered to `i1` signals. Get an lvalue ref to the signal such
    // that we can assign to it.
    auto target = context.convertLvalueExpression(stmt.target);
    if (!target)
      return failure();

    // Read and invert the current value of the signal. Writing this inverted
    // value to the signal is our event signaling mechanism.
    Value inverted = moore::ReadOp::create(builder, loc, target);
    inverted = moore::NotOp::create(builder, loc, inverted);

    if (stmt.isNonBlocking)
      moore::NonBlockingAssignOp::create(builder, loc, target, inverted);
    else
      moore::BlockingAssignOp::create(builder, loc, target, inverted);
    return success();
  }

  /// Emit an error for all other statements.
  template <typename T>
  LogicalResult visit(T &&stmt) {
    mlir::emitError(loc, "unsupported statement: ")
        << slang::ast::toString(stmt.kind);
    return mlir::failure();
  }

  LogicalResult visitInvalid(const slang::ast::Statement &stmt) {
    mlir::emitError(loc, "invalid statement: ")
        << slang::ast::toString(stmt.kind);
    return mlir::failure();
  }
};
} // namespace

LogicalResult Context::convertStatement(const slang::ast::Statement &stmt) {
  assert(builder.getInsertionBlock());
  auto loc = convertLocation(stmt.sourceRange);
  return stmt.visit(StmtVisitor(*this, loc));
}
// NOLINTEND(misc-no-recursion)
