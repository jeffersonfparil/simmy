from array import Array


def main() raises:
    print("Kamusta world!")
    # Vector
    var n: Int = 10
    var data: List[Float64] = [Float64(x) for x in List(range(n))]
    var shape: List[Int] = [n]
    var arr = Array[DType.float64](data, shape)
    print("----------\nVector:")
    print(arr)
    # Matrix
    n = 13
    var p = 9
    data = [Float64(x) for x in List(range(n * p))]
    shape = [n, p]
    arr = Array[DType.float64](data, shape)
    print("----------\nMatrix:")
    print(arr)
    # Tensor (3D)
    n = 10
    p = 27
    var q = 6
    data = [Float64(x) for x in List(range(n * p * q))]
    shape = [n, p, q]
    arr = Array[DType.float64](data, shape)
    print("----------\n3D Tensor:")
    print(arr)
    # Tensor (4D)
    n = 10
    p = 27
    q = 6
    var r = 5
    data = [Float64(x) for x in List(range(n * p * q * r))]
    shape = [n, p, q, r]
    arr = Array[DType.float64](data, shape)
    print("----------\n4D Tensor:")
    print(arr)
    # Tensor (5D)
    n = 10
    p = 27
    q = 6
    r = 5
    var s = 3
    data = [Float64(x) for x in List(range(n * p * q * r * s))]
    shape = [n, p, q, r, s]
    arr = Array[DType.float64](data, shape)
    print("----------\n5D Tensor:")
    print(arr)
