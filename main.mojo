from layout import Layout, LayoutTensor


def main() raises:
    print("Kamusta world!")
    var n: Int = 5
    var p: Int = 4
    var storage = Array[Float32, n * p](uninitialized=True)
    var tensor_5x4 = LayoutTensor[DType.float32, Layout.row_major(5, 4)](storage)