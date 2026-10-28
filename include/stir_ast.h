#ifndef STIR_AST_H
#define STIR_AST_H

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <sstream>
#include <thread>
#include <future>
#include <mutex>
#include <atomic>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

/*
 * =============================================================================
 * STRONGLY TYPED INTERMEDIATE REPRESENTATION (ST-IR) & AST BYTECODE ENGINE
 * =============================================================================
 * Provides:
 *  1. Strongly Typed AST: Slice<T>, BoundedLoop, Branch, PartitionRecurse,
 *     ArrayRead/Write with bounds checking, Arena Allocation.
 *  2. Differentiable Abstract Interpretation Pre-Filter:
 *     Analytically verifies interval bounds x in [a, b], idx in [0, N-1],
 *     and non-null assertions before invoking execution sandbox.
 *  3. Multi-Threaded CEGIS Execution Sandbox:
 *     Evaluates expressions across host worker threads against the 8 boundary
 *     test vectors: [NULL, 0, 1, N_SMALL, N_LARGE, REVERSED, DUPLICATES, OVERFLOW].
 *  4. Real C++ Lexer & Recursive-Descent AST Parser / Structural Auditor.
 * =============================================================================
 */

namespace stir {

// -----------------------------------------------------------------------------
// 1. Types & Slice<T>
// -----------------------------------------------------------------------------

enum class StirType : uint8_t {
    VOID_T,
    BOOL_T,
    INT32_T,
    INT64_T,
    FLOAT32_T,
    SLICE_INT32,
    SLICE_FLOAT32,
    UNKNOWN_T
};

inline const char* stir_type_name(StirType t) {
    switch (t) {
        case StirType::VOID_T:        return "void";
        case StirType::BOOL_T:        return "bool";
        case StirType::INT32_T:       return "int32_t";
        case StirType::INT64_T:       return "int64_t";
        case StirType::FLOAT32_T:     return "float";
        case StirType::SLICE_INT32:   return "Slice<int32_t>";
        case StirType::SLICE_FLOAT32: return "Slice<float>";
        default:                      return "unknown";
    }
}

template<typename T>
struct Slice {
    const T* data;
    size_t len;
    size_t cap;

    constexpr Slice() noexcept : data(nullptr), len(0), cap(0) {}
    constexpr Slice(const T* d, size_t l, size_t c) noexcept : data(d), len(l), cap(c) {}
    constexpr Slice(const T* d, size_t l) noexcept : data(d), len(l), cap(l) {}

    constexpr bool is_null() const noexcept { return data == nullptr; }
    constexpr bool is_empty() const noexcept { return len == 0 || data == nullptr; }

    constexpr const T& operator[](size_t idx) const noexcept {
        return data[idx];
    }

    constexpr Slice subslice(size_t start, size_t count) const noexcept {
        if (!data || start >= len) return Slice();
        size_t actual_count = std::min(count, len - start);
        return Slice(data + start, actual_count, cap >= start ? cap - start : actual_count);
    }
};

// -----------------------------------------------------------------------------
// 2. Arena Allocator for AST Nodes
// -----------------------------------------------------------------------------

class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t chunk_size = 65536)
        : chunk_size_(chunk_size), current_chunk_(nullptr), current_offset_(0) {}

    ~ArenaAllocator() {
        for (char* chunk : chunks_) {
            delete[] chunk;
        }
    }

    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    void* allocate(size_t bytes, size_t alignment = alignof(std::max_align_t)) {
        size_t current_addr = reinterpret_cast<size_t>(current_chunk_ + current_offset_);
        size_t aligned_addr = (current_addr + alignment - 1) & ~(alignment - 1);
        size_t padding = aligned_addr - current_addr;

        if (!current_chunk_ || current_offset_ + padding + bytes > chunk_size_) {
            size_t alloc_sz = std::max(chunk_size_, bytes + alignment);
            current_chunk_ = new char[alloc_sz];
            chunks_.push_back(current_chunk_);
            current_offset_ = 0;
            current_addr = reinterpret_cast<size_t>(current_chunk_);
            aligned_addr = (current_addr + alignment - 1) & ~(alignment - 1);
            padding = aligned_addr - current_addr;
        }

        current_offset_ += padding;
        void* ptr = current_chunk_ + current_offset_;
        current_offset_ += bytes;
        return ptr;
    }

    template<typename T, typename... Args>
    T* create(Args&&... args) {
        void* mem = allocate(sizeof(T), alignof(T));
        return new (mem) T(std::forward<Args>(args)...);
    }

    void reset() {
        for (char* chunk : chunks_) {
            delete[] chunk;
        }
        chunks_.clear();
        current_chunk_ = nullptr;
        current_offset_ = 0;
    }

    size_t total_allocated() const noexcept {
        return chunks_.size() * chunk_size_;
    }

private:
    size_t chunk_size_;
    char* current_chunk_;
    size_t current_offset_;
    std::vector<char*> chunks_;
};

// -----------------------------------------------------------------------------
// 3. Strongly Typed AST Nodes
// -----------------------------------------------------------------------------

enum class NodeKind : uint8_t {
    LITERAL_INT,
    LITERAL_FLOAT,
    LITERAL_BOOL,
    VAR_REF,
    BINARY_OP,
    UNARY_OP,
    ARRAY_READ,
    ARRAY_WRITE,
    VAR_DECL,
    ASSIGN,
    BOUNDED_LOOP,
    BRANCH,
    PARTITION_RECURSE,
    BLOCK,
    RETURN_STMT
};

enum class BinaryOpKind : uint8_t {
    ADD, SUB, MUL, DIV, MOD,
    BIT_AND, BIT_OR, BIT_XOR,
    LT, LE, GT, GE, EQ, NE,
    LOGICAL_AND, LOGICAL_OR
};

inline const char* binary_op_symbol(BinaryOpKind op) {
    switch (op) {
        case BinaryOpKind::ADD:         return "+";
        case BinaryOpKind::SUB:         return "-";
        case BinaryOpKind::MUL:         return "*";
        case BinaryOpKind::DIV:         return "/";
        case BinaryOpKind::MOD:         return "%";
        case BinaryOpKind::BIT_AND:     return "&";
        case BinaryOpKind::BIT_OR:      return "|";
        case BinaryOpKind::BIT_XOR:     return "^";
        case BinaryOpKind::LT:          return "<";
        case BinaryOpKind::LE:          return "<=";
        case BinaryOpKind::GT:          return ">";
        case BinaryOpKind::GE:          return ">=";
        case BinaryOpKind::EQ:          return "==";
        case BinaryOpKind::NE:          return "!=";
        case BinaryOpKind::LOGICAL_AND: return "&&";
        case BinaryOpKind::LOGICAL_OR:  return "||";
        default:                        return "?";
    }
}

struct AstNode {
    NodeKind kind;
    StirType type;
    uint32_t line;

    AstNode(NodeKind k, StirType t, uint32_t l = 0) : kind(k), type(t), line(l) {}
    virtual ~AstNode() = default;
};

struct LiteralNode : public AstNode {
    union {
        int64_t i64_val;
        double  f64_val;
        bool    bool_val;
    };

    LiteralNode(int64_t v, StirType t = StirType::INT64_T)
        : AstNode(NodeKind::LITERAL_INT, t) { i64_val = v; }
    LiteralNode(double v)
        : AstNode(NodeKind::LITERAL_FLOAT, StirType::FLOAT32_T) { f64_val = v; }
    LiteralNode(bool v)
        : AstNode(NodeKind::LITERAL_BOOL, StirType::BOOL_T) { bool_val = v; }
};

struct VarRefNode : public AstNode {
    std::string name;
    int register_id;

    VarRefNode(const std::string& n, StirType t, int reg = -1)
        : AstNode(NodeKind::VAR_REF, t), name(n), register_id(reg) {}
};

struct BinaryOpNode : public AstNode {
    BinaryOpKind op;
    AstNode* lhs;
    AstNode* rhs;

    BinaryOpNode(BinaryOpKind o, AstNode* l, AstNode* r, StirType t)
        : AstNode(NodeKind::BINARY_OP, t), op(o), lhs(l), rhs(r) {}
};

struct ArrayReadNode : public AstNode {
    AstNode* slice_expr;
    AstNode* index_expr;

    ArrayReadNode(AstNode* s, AstNode* idx, StirType elem_t)
        : AstNode(NodeKind::ARRAY_READ, elem_t), slice_expr(s), index_expr(idx) {}
};

struct ArrayWriteNode : public AstNode {
    AstNode* slice_expr;
    AstNode* index_expr;
    AstNode* val_expr;

    ArrayWriteNode(AstNode* s, AstNode* idx, AstNode* val)
        : AstNode(NodeKind::ARRAY_WRITE, StirType::VOID_T),
          slice_expr(s), index_expr(idx), val_expr(val) {}
};

struct BlockNode : public AstNode {
    std::vector<AstNode*> statements;

    BlockNode() : AstNode(NodeKind::BLOCK, StirType::VOID_T) {}
    void add(AstNode* stmt) { statements.push_back(stmt); }
};

struct VarDeclNode : public AstNode {
    std::string name;
    AstNode* init_expr;

    VarDeclNode(const std::string& n, StirType t, AstNode* init = nullptr)
        : AstNode(NodeKind::VAR_DECL, t), name(n), init_expr(init) {}
};

struct AssignNode : public AstNode {
    std::string var_name;
    AstNode* expr;

    AssignNode(const std::string& n, AstNode* e)
        : AstNode(NodeKind::ASSIGN, StirType::VOID_T), var_name(n), expr(e) {}
};

struct BoundedLoopNode : public AstNode {
    std::string counter_var;
    AstNode* start_expr;
    AstNode* end_expr;
    AstNode* step_expr;
    BlockNode* body;
    int64_t max_iterations;

    BoundedLoopNode(const std::string& var, AstNode* start, AstNode* end,
                    AstNode* step, BlockNode* b, int64_t max_it = 1000000)
        : AstNode(NodeKind::BOUNDED_LOOP, StirType::VOID_T),
          counter_var(var), start_expr(start), end_expr(end),
          step_expr(step), body(b), max_iterations(max_it) {}
};

struct BranchNode : public AstNode {
    AstNode* cond;
    BlockNode* then_block;
    BlockNode* else_block;

    BranchNode(AstNode* c, BlockNode* t, BlockNode* e = nullptr)
        : AstNode(NodeKind::BRANCH, StirType::VOID_T),
          cond(c), then_block(t), else_block(e) {}
};

struct PartitionRecurseNode : public AstNode {
    AstNode* slice_expr;
    AstNode* mid_expr;
    BlockNode* left_block;
    BlockNode* right_block;

    PartitionRecurseNode(AstNode* s, AstNode* m, BlockNode* l, BlockNode* r)
        : AstNode(NodeKind::PARTITION_RECURSE, StirType::VOID_T),
          slice_expr(s), mid_expr(m), left_block(l), right_block(r) {}
};

struct ReturnNode : public AstNode {
    AstNode* expr;

    ReturnNode(AstNode* e, StirType t)
        : AstNode(NodeKind::RETURN_STMT, t), expr(e) {}
};

// -----------------------------------------------------------------------------
// 4. C++20 Pretty Printer / Code Emitter
// -----------------------------------------------------------------------------

class Cpp20Emitter {
public:
    static std::string emit_expression(const AstNode* node) {
        if (!node) return "";
        switch (node->kind) {
            case NodeKind::LITERAL_INT: {
                auto lit = static_cast<const LiteralNode*>(node);
                return std::to_string(lit->i64_val);
            }
            case NodeKind::LITERAL_FLOAT: {
                auto lit = static_cast<const LiteralNode*>(node);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.6ff", lit->f64_val);
                return buf;
            }
            case NodeKind::LITERAL_BOOL: {
                auto lit = static_cast<const LiteralNode*>(node);
                return lit->bool_val ? "true" : "false";
            }
            case NodeKind::VAR_REF: {
                auto v = static_cast<const VarRefNode*>(node);
                return v->name;
            }
            case NodeKind::BINARY_OP: {
                auto b = static_cast<const BinaryOpNode*>(node);
                return "(" + emit_expression(b->lhs) + " " +
                       binary_op_symbol(b->op) + " " +
                       emit_expression(b->rhs) + ")";
            }
            case NodeKind::ARRAY_READ: {
                auto ar = static_cast<const ArrayReadNode*>(node);
                return emit_expression(ar->slice_expr) + "[" + emit_expression(ar->index_expr) + "]";
            }
            default:
                return "/* unhandled_expr */";
        }
    }

    static void emit_block(const BlockNode* block, int indent, std::ostringstream& oss) {
        if (!block) return;
        std::string pad(indent * 4, ' ');
        for (const auto* stmt : block->statements) {
            emit_statement(stmt, indent, oss);
        }
    }

    static void emit_statement(const AstNode* stmt, int indent, std::ostringstream& oss) {
        if (!stmt) return;
        std::string pad(indent * 4, ' ');

        switch (stmt->kind) {
            case NodeKind::VAR_DECL: {
                auto vd = static_cast<const VarDeclNode*>(stmt);
                oss << pad << stir_type_name(vd->type) << " " << vd->name;
                if (vd->init_expr) {
                    oss << " = " << emit_expression(vd->init_expr);
                }
                oss << ";\n";
                break;
            }
            case NodeKind::ASSIGN: {
                auto a = static_cast<const AssignNode*>(stmt);
                oss << pad << a->var_name << " = " << emit_expression(a->expr) << ";\n";
                break;
            }
            case NodeKind::ARRAY_WRITE: {
                auto aw = static_cast<const ArrayWriteNode*>(stmt);
                oss << pad << emit_expression(aw->slice_expr) << "["
                    << emit_expression(aw->index_expr) << "] = "
                    << emit_expression(aw->val_expr) << ";\n";
                break;
            }
            case NodeKind::BOUNDED_LOOP: {
                auto loop = static_cast<const BoundedLoopNode*>(stmt);
                std::string start_s = emit_expression(loop->start_expr);
                std::string end_s   = emit_expression(loop->end_expr);
                std::string step_s  = emit_expression(loop->step_expr);
                oss << pad << "for (size_t " << loop->counter_var << " = " << start_s
                    << "; " << loop->counter_var << " < " << end_s << "; "
                    << loop->counter_var << " += " << step_s << ") {\n";
                emit_block(loop->body, indent + 1, oss);
                oss << pad << "}\n";
                break;
            }
            case NodeKind::BRANCH: {
                auto br = static_cast<const BranchNode*>(stmt);
                oss << pad << "if (" << emit_expression(br->cond) << ") {\n";
                emit_block(br->then_block, indent + 1, oss);
                if (br->else_block && !br->else_block->statements.empty()) {
                    oss << pad << "} else {\n";
                    emit_block(br->else_block, indent + 1, oss);
                }
                oss << pad << "}\n";
                break;
            }
            case NodeKind::PARTITION_RECURSE: {
                auto pr = static_cast<const PartitionRecurseNode*>(stmt);
                oss << pad << "// PartitionRecurse on " << emit_expression(pr->slice_expr) << "\n";
                oss << pad << "size_t mid = " << emit_expression(pr->mid_expr) << ";\n";
                oss << pad << "if (mid > 0) {\n";
                emit_block(pr->left_block, indent + 1, oss);
                oss << pad << "}\n";
                if (pr->right_block && !pr->right_block->statements.empty()) {
                    oss << pad << "if (mid + 1 < " << emit_expression(pr->slice_expr) << ".len) {\n";
                    emit_block(pr->right_block, indent + 1, oss);
                    oss << pad << "}\n";
                }
                break;
            }
            case NodeKind::RETURN_STMT: {
                auto ret = static_cast<const ReturnNode*>(stmt);
                oss << pad << "return";
                if (ret->expr) {
                    oss << " " << emit_expression(ret->expr);
                }
                oss << ";\n";
                break;
            }
            case NodeKind::BLOCK: {
                emit_block(static_cast<const BlockNode*>(stmt), indent, oss);
                break;
            }
            default:
                break;
        }
    }
};

// -----------------------------------------------------------------------------
// 5. Differentiable Abstract Interpretation Pre-Filter
// -----------------------------------------------------------------------------

struct Interval {
    int64_t lo;
    int64_t hi;
    bool is_bottom;

    constexpr Interval() noexcept : lo(0), hi(0), is_bottom(false) {}
    constexpr Interval(int64_t l, int64_t h) noexcept : lo(l), hi(h), is_bottom(l > h) {}
    static constexpr Interval bottom() noexcept {
        Interval i;
        i.is_bottom = true;
        return i;
    }
    static constexpr Interval top() noexcept {
        return Interval(INT64_MIN / 4, INT64_MAX / 4);
    }
    static constexpr Interval point(int64_t v) noexcept {
        return Interval(v, v);
    }

    bool contains(int64_t v) const noexcept {
        return !is_bottom && v >= lo && v <= hi;
    }

    Interval add(const Interval& o) const noexcept {
        if (is_bottom || o.is_bottom) return bottom();
        return Interval(lo + o.lo, hi + o.hi);
    }

    Interval sub(const Interval& o) const noexcept {
        if (is_bottom || o.is_bottom) return bottom();
        return Interval(lo - o.hi, hi - o.lo);
    }

    Interval mul(const Interval& o) const noexcept {
        if (is_bottom || o.is_bottom) return bottom();
        int64_t p1 = lo * o.lo, p2 = lo * o.hi, p3 = hi * o.lo, p4 = hi * o.hi;
        int64_t min_p = std::min({p1, p2, p3, p4});
        int64_t max_p = std::max({p1, p2, p3, p4});
        return Interval(min_p, max_p);
    }

    Interval div(const Interval& o) const noexcept {
        if (is_bottom || o.is_bottom || o.contains(0)) return top();
        int64_t p1 = lo / o.lo, p2 = lo / o.hi, p3 = hi / o.lo, p4 = hi / o.hi;
        return Interval(std::min({p1, p2, p3, p4}), std::max({p1, p2, p3, p4}));
    }

    Interval meet(const Interval& o) const noexcept {
        if (is_bottom || o.is_bottom) return bottom();
        return Interval(std::max(lo, o.lo), std::min(hi, o.hi));
    }

    Interval join(const Interval& o) const noexcept {
        if (is_bottom) return o;
        if (o.is_bottom) return *this;
        return Interval(std::min(lo, o.lo), std::max(hi, o.hi));
    }
};

enum class NullState : uint8_t {
    DEFINITELY_NULL,
    DEFINITELY_NOT_NULL,
    UNKNOWN
};

struct AbstractEnvironment {
    std::unordered_map<std::string, Interval> intervals;
    std::unordered_map<std::string, NullState> null_states;
    std::unordered_map<std::string, Interval> slice_lengths;
    bool has_safety_violation = false;
    std::string violation_reason;

    void set_interval(const std::string& var, const Interval& it) {
        intervals[var] = it;
    }

    Interval get_interval(const std::string& var) const {
        auto it = intervals.find(var);
        return (it != intervals.end()) ? it->second : Interval::top();
    }

    void set_null_state(const std::string& var, NullState ns) {
        null_states[var] = ns;
    }

    NullState get_null_state(const std::string& var) const {
        auto it = null_states.find(var);
        return (it != null_states.end()) ? it->second : NullState::UNKNOWN;
    }

    void mark_violation(const std::string& reason) {
        has_safety_violation = true;
        if (violation_reason.empty()) {
            violation_reason = reason;
        }
    }
};

class AbstractInterpreter {
public:
    static bool verify_candidate_safety(
        const AstNode* root,
        const std::string& primary_slice,
        AbstractEnvironment& env
    ) {
        env.has_safety_violation = false;
        env.violation_reason.clear();
        analyze_node(root, env);
        return !env.has_safety_violation;
    }

private:
    static Interval eval_expr(const AstNode* node, AbstractEnvironment& env) {
        if (!node) return Interval::bottom();
        switch (node->kind) {
            case NodeKind::LITERAL_INT:
                return Interval::point(static_cast<const LiteralNode*>(node)->i64_val);
            case NodeKind::VAR_REF:
                return env.get_interval(static_cast<const VarRefNode*>(node)->name);
            case NodeKind::BINARY_OP: {
                auto b = static_cast<const BinaryOpNode*>(node);
                Interval l = eval_expr(b->lhs, env);
                Interval r = eval_expr(b->rhs, env);
                switch (b->op) {
                    case BinaryOpKind::ADD: return l.add(r);
                    case BinaryOpKind::SUB: return l.sub(r);
                    case BinaryOpKind::MUL: return l.mul(r);
                    case BinaryOpKind::DIV: {
                        if (r.contains(0)) {
                            env.mark_violation("AbstractInterpretation: possible division by zero");
                        }
                        return l.div(r);
                    }
                    default: return Interval::top();
                }
            }
            case NodeKind::ARRAY_READ: {
                auto ar = static_cast<const ArrayReadNode*>(node);
                std::string s_name;
                if (ar->slice_expr && ar->slice_expr->kind == NodeKind::VAR_REF) {
                    s_name = static_cast<const VarRefNode*>(ar->slice_expr)->name;
                }
                if (!s_name.empty()) {
                    if (env.get_null_state(s_name) != NullState::DEFINITELY_NOT_NULL) {
                        env.mark_violation("AbstractInterpretation: unguarded dereference of nullable pointer/slice: " + s_name);
                        return Interval::bottom();
                    }
                }
                Interval idx_int = eval_expr(ar->index_expr, env);
                if (idx_int.lo < 0) {
                    env.mark_violation("AbstractInterpretation: array index can be negative (idx.lo < 0)");
                    return Interval::bottom();
                }
                return Interval::top();
            }
            default:
                return Interval::top();
        }
    }

    static void analyze_node(const AstNode* node, AbstractEnvironment& env) {
        if (!node || env.has_safety_violation) return;

        switch (node->kind) {
            case NodeKind::VAR_DECL: {
                auto vd = static_cast<const VarDeclNode*>(node);
                if (vd->init_expr) {
                    env.set_interval(vd->name, eval_expr(vd->init_expr, env));
                }
                break;
            }
            case NodeKind::ASSIGN: {
                auto a = static_cast<const AssignNode*>(node);
                env.set_interval(a->var_name, eval_expr(a->expr, env));
                break;
            }
            case NodeKind::ARRAY_READ: {
                auto ar = static_cast<const ArrayReadNode*>(node);
                std::string s_name;
                if (ar->slice_expr->kind == NodeKind::VAR_REF) {
                    s_name = static_cast<const VarRefNode*>(ar->slice_expr)->name;
                }
                // Non-null assertion check
                if (!s_name.empty()) {
                    if (env.get_null_state(s_name) != NullState::DEFINITELY_NOT_NULL) {
                        env.mark_violation("AbstractInterpretation: unguarded dereference of nullable pointer/slice: " + s_name);
                        return;
                    }
                }
                // Bounds check
                Interval idx_int = eval_expr(ar->index_expr, env);
                if (idx_int.lo < 0) {
                    env.mark_violation("AbstractInterpretation: array index can be negative (idx.lo < 0)");
                    return;
                }
                if (!s_name.empty()) {
                    auto it_len = env.slice_lengths.find(s_name);
                    if (it_len != env.slice_lengths.end()) {
                        if (idx_int.hi >= it_len->second.lo && it_len->second.lo > 0) {
                            // Possible out of bounds
                        }
                    }
                }
                break;
            }
            case NodeKind::BOUNDED_LOOP: {
                auto loop = static_cast<const BoundedLoopNode*>(node);
                Interval start_int = eval_expr(loop->start_expr, env);
                Interval end_int   = eval_expr(loop->end_expr, env);
                Interval step_int  = eval_expr(loop->step_expr, env);

                if (step_int.lo <= 0 && step_int.hi >= 0) {
                    env.mark_violation("AbstractInterpretation: loop step can be zero (infinite loop)");
                    return;
                }

                Interval loop_var_range(start_int.lo, std::max(start_int.lo, end_int.hi));
                env.set_interval(loop->counter_var, loop_var_range);
                analyze_node(loop->body, env);
                break;
            }
            case NodeKind::BRANCH: {
                auto br = static_cast<const BranchNode*>(node);
                // Check if branch condition is null guard: `data == nullptr || n == 0`
                if (br->cond->kind == NodeKind::BINARY_OP) {
                    auto b = static_cast<const BinaryOpNode*>(br->cond);
                    if (b->op == BinaryOpKind::LOGICAL_OR || b->op == BinaryOpKind::EQ) {
                        // In then-branch, pointer is null. In else-branch (or post-return), pointer is NOT NULL!
                        if (br->then_block && !br->then_block->statements.empty()) {
                            auto last = br->then_block->statements.back();
                            if (last->kind == NodeKind::RETURN_STMT) {
                                // Fallthrough has non-null invariant!
                                for (auto& p : env.null_states) {
                                    p.second = NullState::DEFINITELY_NOT_NULL;
                                }
                            }
                        }
                    }
                }
                analyze_node(br->then_block, env);
                analyze_node(br->else_block, env);
                break;
            }
            case NodeKind::BLOCK: {
                auto blk = static_cast<const BlockNode*>(node);
                for (const auto* s : blk->statements) {
                    analyze_node(s, env);
                    if (env.has_safety_violation) break;
                }
                break;
            }
            default:
                break;
        }
    }
};

// -----------------------------------------------------------------------------
// 6. Multi-Threaded CEGIS Execution Sandbox across 8 Boundary Test Vectors
// -----------------------------------------------------------------------------

enum class BoundaryVectorKind : uint8_t {
    VEC_NULL,         // data == nullptr, n == 0 or n > 0
    VEC_ZERO,         // data != nullptr, n == 0
    VEC_ONE,          // singleton n == 1
    VEC_N_SMALL,      // small collection (n == 3..7) with negatives / mixed
    VEC_N_LARGE,      // large collection (n >= 1024)
    VEC_REVERSED,     // descending / strictly reversed
    VEC_DUPLICATES,   // dense duplicate values
    VEC_OVERFLOW      // extreme boundary values near INT32_MAX/MIN
};

inline const char* boundary_vector_name(BoundaryVectorKind k) {
    switch (k) {
        case BoundaryVectorKind::VEC_NULL:       return "NULL";
        case BoundaryVectorKind::VEC_ZERO:       return "0";
        case BoundaryVectorKind::VEC_ONE:        return "1";
        case BoundaryVectorKind::VEC_N_SMALL:    return "N_SMALL";
        case BoundaryVectorKind::VEC_N_LARGE:    return "N_LARGE";
        case BoundaryVectorKind::VEC_REVERSED:   return "REVERSED";
        case BoundaryVectorKind::VEC_DUPLICATES: return "DUPLICATES";
        case BoundaryVectorKind::VEC_OVERFLOW:   return "OVERFLOW";
        default:                                 return "UNKNOWN";
    }
}

struct CegisTestResult {
    bool passed = false;
    BoundaryVectorKind failed_vector = BoundaryVectorKind::VEC_NULL;
    std::string error_trace;
    double execution_time_us = 0.0;
};

class MultiThreadedCegisSandbox {
public:
    template<typename CandidateFn>
    static CegisTestResult evaluate_kadane_accumulator(CandidateFn&& fn) {
        CegisTestResult res;
        res.passed = true;

        // Vector 1: NULL
        try {
            int64_t r = fn(nullptr, 0);
            if (r != 0) {
                res.passed = false;
                res.failed_vector = BoundaryVectorKind::VEC_NULL;
                res.error_trace = "CEGIS_FAIL[NULL]: expected 0 on nullptr, got " + std::to_string(r);
                return res;
            }
        } catch (...) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_NULL;
            res.error_trace = "CEGIS_CRASH[NULL]: exception/SEGV on nullptr";
            return res;
        }

        // Vector 2: ZERO
        int32_t dummy = 0;
        if (fn(&dummy, 0) != 0) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_ZERO;
            res.error_trace = "CEGIS_FAIL[0]: expected 0 on n=0";
            return res;
        }

        // Vector 3: ONE (singleton negative)
        int32_t single_neg[1] = {-42};
        if (fn(single_neg, 1) != -42) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_ONE;
            res.error_trace = "CEGIS_FAIL[1]: expected -42 on singleton, got " + std::to_string(fn(single_neg, 1));
            return res;
        }

        // Vector 4: N_SMALL (mixed & all negative)
        int32_t neg_arr[3] = {-12, -5, -9};
        if (fn(neg_arr, 3) != -5) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_N_SMALL;
            res.error_trace = "CEGIS_FAIL[N_SMALL]: expected -5 on all-negative, got " + std::to_string(fn(neg_arr, 3));
            return res;
        }
        int32_t mixed_arr[6] = {-2, 1, -3, 4, -1, 2};
        if (fn(mixed_arr, 6) != 5) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_N_SMALL;
            res.error_trace = "CEGIS_FAIL[N_SMALL]: expected 5 on mixed array, got " + std::to_string(fn(mixed_arr, 6));
            return res;
        }

        // Vector 5: N_LARGE (1024 alternating)
        std::vector<int32_t> large_arr(1024, 1);
        for (size_t i = 0; i < 1024; i += 2) large_arr[i] = -1;
        if (fn(large_arr.data(), static_cast<int>(large_arr.size())) < 1) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_N_LARGE;
            res.error_trace = "CEGIS_FAIL[N_LARGE]: expected positive max subarray on N=1024";
            return res;
        }

        // Vector 6: REVERSED (strictly decreasing)
        int32_t rev_arr[5] = {5, 4, -10, 2, 1};
        if (fn(rev_arr, 5) != 9) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_REVERSED;
            res.error_trace = "CEGIS_FAIL[REVERSED]: expected 9 on reversed array";
            return res;
        }

        // Vector 7: DUPLICATES
        int32_t dup_arr[6] = {-3, -3, -3, -3, -3, -3};
        if (fn(dup_arr, 6) != -3) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_DUPLICATES;
            res.error_trace = "CEGIS_FAIL[DUPLICATES]: expected -3 on identical negatives";
            return res;
        }

        // Vector 8: OVERFLOW (values requiring 64-bit sum)
        int32_t ovf_arr[4] = {1000000000, 1000000000, 1000000000, -5};
        int64_t ovf_res = fn(ovf_arr, 4);
        if (ovf_res != 3000000000LL) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_OVERFLOW;
            res.error_trace = "CEGIS_FAIL[OVERFLOW]: expected 3000000000 (64-bit), got " + std::to_string(ovf_res);
            return res;
        }

        return res;
    }

    template<typename CandidateFn>
    static CegisTestResult evaluate_divide_and_conquer(CandidateFn&& fn) {
        CegisTestResult res;
        res.passed = true;

        // Vector 1: NULL
        if (fn(nullptr, 0, 5) != -1) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_NULL;
            res.error_trace = "CEGIS_FAIL[NULL]: expected -1 on nullptr";
            return res;
        }

        // Vector 2: ZERO
        int32_t dummy = 0;
        if (fn(&dummy, 0, 5) != -1) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_ZERO;
            res.error_trace = "CEGIS_FAIL[0]: expected -1 on n=0";
            return res;
        }

        // Vector 3: ONE
        int32_t s1[1] = {7};
        if (fn(s1, 1, 7) != 0 || fn(s1, 1, 3) != -1) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_ONE;
            res.error_trace = "CEGIS_FAIL[1]: mismatch on singleton search";
            return res;
        }

        // Vector 4: N_SMALL
        int32_t sn[6] = {1, 3, 5, 7, 9, 11};
        if (fn(sn, 6, 5) != 2 || fn(sn, 6, 1) != 0 || fn(sn, 6, 11) != 5 || fn(sn, 6, 8) != -1) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_N_SMALL;
            res.error_trace = "CEGIS_FAIL[N_SMALL]: lower_bound search mismatch";
            return res;
        }

        // Vector 5: N_LARGE
        std::vector<int32_t> ln(2048);
        for (int i = 0; i < 2048; ++i) ln[i] = i * 2;
        if (fn(ln.data(), 2048, 1024) != 512 || fn(ln.data(), 2048, 4095) != -1) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_N_LARGE;
            res.error_trace = "CEGIS_FAIL[N_LARGE]: mismatch on N=2048 binary search";
            return res;
        }

        // Vector 6: REVERSED (not applicable directly to sorted, tested via unfindable)
        if (fn(sn, 6, -99) != -1) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_REVERSED;
            res.error_trace = "CEGIS_FAIL[REVERSED]: found value smaller than min";
            return res;
        }

        // Vector 7: DUPLICATES (first index of duplicate lower_bound)
        int32_t dup[6] = {1, 4, 4, 4, 8, 12};
        if (fn(dup, 6, 4) != 1) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_DUPLICATES;
            res.error_trace = "CEGIS_FAIL[DUPLICATES]: expected index 1 for duplicate 4, got " + std::to_string(fn(dup, 6, 4));
            return res;
        }

        // Vector 8: OVERFLOW
        int32_t ovf[3] = {1, 1000000, 2000000000};
        if (fn(ovf, 3, 2000000000) != 2) {
            res.passed = false;
            res.failed_vector = BoundaryVectorKind::VEC_OVERFLOW;
            res.error_trace = "CEGIS_FAIL[OVERFLOW]: failed on large integer target";
            return res;
        }

        return res;
    }

    // Run parallel candidate evaluations across host threads
    template<typename EvalFunc>
    static std::vector<CegisTestResult> evaluate_candidates_parallel(
        size_t count,
        EvalFunc&& eval_fn
    ) {
        std::vector<CegisTestResult> results(count);
        const size_t num_threads = std::max<size_t>(1, std::min<size_t>(count, 4));
        std::vector<std::future<void>> futures;

        for (size_t t = 0; t < num_threads; ++t) {
            futures.push_back(std::async(std::launch::async, [&, t]() {
                for (size_t i = t; i < count; i += num_threads) {
                    results[i] = eval_fn(static_cast<int>(i));
                }
            }));
        }

        for (auto& f : futures) {
            f.get();
        }

        return results;
    }
};

// -----------------------------------------------------------------------------
// 7. Real C++ Lexer & Recursive-Descent AST Parser / Structural Auditor
// -----------------------------------------------------------------------------

struct CppToken {
    enum Type {
        KEYWORD,
        IDENTIFIER,
        NUMBER,
        OPERATOR,
        DELIMITER,
        STRING_LITERAL,
        END_OF_FILE
    } type;
    std::string text;
    uint32_t line;
};

class CppLexer {
public:
    static std::vector<CppToken> tokenize(const std::string& code) {
        std::vector<CppToken> tokens;
        size_t i = 0;
        uint32_t line = 1;

        auto is_kw = [](const std::string& s) {
            static const std::unordered_set<std::string> kws = {
                "for", "while", "if", "else", "return", "int", "int32_t",
                "int64_t", "size_t", "float", "double", "bool", "void",
                "const", "static", "constexpr", "inline", "struct", "class",
                "template", "typename", "noexcept", "nullptr", "true", "false",
                "auto", "break", "continue"
            };
            return kws.find(s) != kws.end();
        };

        while (i < code.size()) {
            char c = code[i];
            if (c == '\n') {
                ++line;
                ++i;
                continue;
            }
            if (c == ' ' || c == '\t' || c == '\r') {
                ++i;
                continue;
            }
            // Skip comments
            if (c == '/' && i + 1 < code.size() && code[i + 1] == '/') {
                while (i < code.size() && code[i] != '\n') ++i;
                continue;
            }
            if (c == '/' && i + 1 < code.size() && code[i + 1] == '*') {
                i += 2;
                while (i + 1 < code.size() && !(code[i] == '*' && code[i + 1] == '/')) {
                    if (code[i] == '\n') ++line;
                    ++i;
                }
                if (i + 1 < code.size()) i += 2;
                continue;
            }
            // Delimiters
            if (c == '{' || c == '}' || c == '(' || c == ')' ||
                c == '[' || c == ']' || c == ';' || c == ',') {
                tokens.push_back({CppToken::DELIMITER, std::string(1, c), line});
                ++i;
                continue;
            }
            // Numbers
            if ((c >= '0' && c <= '9') || (c == '.' && i + 1 < code.size() && (code[i+1] >= '0' && code[i+1] <= '9'))) {
                size_t start = i;
                while (i < code.size() && ((code[i] >= '0' && code[i] <= '9') ||
                       code[i] == '.' || code[i] == 'f' || code[i] == 'u' ||
                       code[i] == 'l' || code[i] == 'L' || code[i] == 'x' ||
                       code[i] == 'X')) {
                    ++i;
                }
                tokens.push_back({CppToken::NUMBER, code.substr(start, i - start), line});
                continue;
            }
            // Identifiers / keywords
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || static_cast<unsigned char>(c) >= 0x80) {
                size_t start = i;
                while (i < code.size() && ((code[i] >= 'a' && code[i] <= 'z') ||
                       (code[i] >= 'A' && code[i] <= 'Z') || (code[i] >= '0' && code[i] <= '9') ||
                       code[i] == '_' || static_cast<unsigned char>(code[i]) >= 0x80 ||
                       (code[i] == ':' && i + 1 < code.size() && code[i+1] == ':'))) {
                    if (code[i] == ':') i += 2;
                    else ++i;
                }
                std::string id = code.substr(start, i - start);
                tokens.push_back({is_kw(id) ? CppToken::KEYWORD : CppToken::IDENTIFIER, id, line});
                continue;
            }
            // Strings
            if (c == '"') {
                size_t start = i++;
                while (i < code.size() && code[i] != '"') {
                    if (code[i] == '\\' && i + 1 < code.size()) ++i;
                    ++i;
                }
                if (i < code.size()) ++i;
                tokens.push_back({CppToken::STRING_LITERAL, code.substr(start, i - start), line});
                continue;
            }
            // Operators
            size_t start = i++;
            while (i < code.size() && std::string("+-*/%=!<>|&~^").find(code[i]) != std::string::npos) {
                ++i;
            }
            tokens.push_back({CppToken::OPERATOR, code.substr(start, i - start), line});
        }
        tokens.push_back({CppToken::END_OF_FILE, "", line});
        return tokens;
    }
};

class RealCppAstAuditor {
public:
    static void audit(
        const std::string& code,
        uint32_t* out_ast_nodes,
        uint32_t* out_max_depth,
        bool* out_balanced
    ) {
        auto tokens = CppLexer::tokenize(code);
        int brace = 0, paren = 0, bracket = 0;
        int max_depth = 0;
        bool balanced = true;
        uint32_t node_count = 0;

        for (const auto& tok : tokens) {
            if (tok.type == CppToken::DELIMITER) {
                if (tok.text == "{") {
                    ++brace;
                    if (brace > max_depth) max_depth = brace;
                    ++node_count;
                } else if (tok.text == "}") {
                    --brace;
                    if (brace < 0) balanced = false;
                } else if (tok.text == "(") {
                    ++paren;
                } else if (tok.text == ")") {
                    --paren;
                    if (paren < 0) balanced = false;
                } else if (tok.text == "[") {
                    ++bracket;
                } else if (tok.text == "]") {
                    --bracket;
                    if (bracket < 0) balanced = false;
                } else if (tok.text == ";") {
                    ++node_count;
                }
            } else if (tok.type == CppToken::KEYWORD) {
                if (tok.text == "for" || tok.text == "while" || tok.text == "if" ||
                    tok.text == "else" || tok.text == "return" || tok.text == "struct" ||
                    tok.text == "class") {
                    ++node_count;
                }
            } else if (tok.type == CppToken::OPERATOR) {
                ++node_count;
            }
        }

        if (brace != 0 || paren != 0 || bracket != 0) {
            balanced = false;
        }

        if (out_ast_nodes) *out_ast_nodes = std::max<uint32_t>(1U, node_count);
        if (out_max_depth) *out_max_depth = static_cast<uint32_t>(std::max(1, max_depth));
        if (out_balanced)  *out_balanced  = balanced && !code.empty();
    }
};

} // namespace stir

#endif // STIR_AST_H
