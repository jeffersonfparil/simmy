// Unary Operations
const OP_ABS  : u32 = 0u;
const OP_NEG  : u32 = 1u;
const OP_SQRT : u32 = 2u;
const OP_EXP  : u32 = 3u;
const OP_LOG  : u32 = 4u;
const OP_SIN  : u32 = 5u;
const OP_COS  : u32 = 6u;
const OP_NOT  : u32 = 7u; // logical (not bitwise)

// Binary operations
const OP_ADD   : u32 = 0u;
const OP_SUB   : u32 = 1u;
const OP_MUL   : u32 = 2u;
const OP_DIV   : u32 = 3u;
const OP_MIN   : u32 = 4u;
const OP_MAX   : u32 = 5u;
const OP_POW   : u32 = 6u;
const OP_ATAN2 : u32 = 7u;
const OP_EQ    : u32 = 8u;
const OP_NE    : u32 = 9u;
const OP_LT    : u32 = 10u;
const OP_LE    : u32 = 11u;
const OP_GT    : u32 = 12u;
const OP_GE    : u32 = 13u;
const OP_AND   : u32 = 14u; // logical (not bitwise)
const OP_OR    : u32 = 15u; // logical (not bitwise)
const OP_XOR   : u32 = 16u; // logical (not bitwise)

// Contraction: pairwise operations
const OP_PAIR_ADD : u32 = 0u;
const OP_PAIR_SUB : u32 = 1u;
const OP_PAIR_MUL : u32 = 2u;
const OP_PAIR_DIV : u32 = 3u;
const OP_PAIR_MIN : u32 = 4u;
const OP_PAIR_MAX : u32 = 5u;
const OP_PAIR_EQ  : u32 = 6u;
const OP_PAIR_NE  : u32 = 7u;
const OP_PAIR_LT  : u32 = 8u;
const OP_PAIR_LE  : u32 = 9u;
const OP_PAIR_GT  : u32 = 10u;
const OP_PAIR_GE  : u32 = 11u;
const OP_PAIR_AND : u32 = 12u; // logical (not bitwise)
const OP_PAIR_OR  : u32 = 13u; // logical (not bitwise)
const OP_PAIR_XOR : u32 = 14u; // logical (not bitwise)

// Contraction: reduction operations
const OP_REDUCE_ADD : u32 = 0u;
const OP_REDUCE_MUL : u32 = 1u;
const OP_REDUCE_MIN : u32 = 2u;
const OP_REDUCE_MAX : u32 = 3u;
const OP_REDUCE_AND : u32 = 4u; // logical (not bitwise)
const OP_REDUCE_OR  : u32 = 5u; // logical (not bitwise)
const OP_REDUCE_XOR : u32 = 6u; // logical (not bitwise)
