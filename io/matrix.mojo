@fieldwise_init
struct Matrix(Movable):
    var rows: UInt
    var cols: UInt
    vat data: List[List[Float32]]