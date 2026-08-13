struct Array[dtype: DType = DType.float64](Movable, Writable):
    """
    Array type which can be used to define vectors, matrices, and tensors, in general.
    
    # Parameter
        dtype: Data type of the elements. Defaults to Dtype.float64.
    """
    var data: List[Scalar[Self.dtype]]
    """Data is stored contiguously in the heap."""
    var shape: List[Int]
    """
    The size of each dimension.
    # Examples:
    - vector: [4]
    - matrix: [2, 3]
    - 3D tensor: [2, 4, 7].
    """
    var strides: List[Int]
    """
    Defines how the data is structured across dimensions.
    # Examples: 
    - vector: [1]
    - row-major matrix: [7, 1]
    - column-major matrix: [1, 7]
    - 3rd-dimension-minor 3D tensor: [15, 3, 1].
    """

    def __init__(out self, data: List[Scalar[Self.dtype]], shape: List[Int], *, minor_dimension: Int = -1 ) raises:
        var n: Int = len(data)
        var m: Int = 1
        for s in shape:
            m *= s
            if m == 0:
                break
        if n != m:
            raise Error("Invalid shape (" + String(shape) + ") for data of length " + String(n) + "!")
        var d: Int = len(shape)
        if (minor_dimension > d - 1) or (minor_dimension < -d):
            raise Error("Invalid minor dimension for a " + String(d) + "-dimensional array!")
        var minor_d = d + minor_dimension if minor_dimension < 0 else minor_dimension
        # To find the strides given the minor dimenion:
        # - First, we find the products of the dimensions sizes starting from the minor dimension which starts at 1 always
        var products = List[Int](length=d, fill=1)
        for i, j in enumerate(range(minor_d+1, d)):
            products[i+1] = products[i] * shape[j]
        for i_tmp, j in enumerate(range(minor_d)):
            var i = i_tmp + (d - minor_d)
            products[i] = products[i-1] * shape[j]
        # - Finally, we place the dimension size products to their corresponding stride locations
        var strides = List[Int](length=d, fill=1)
        for i, j in enumerate(range(minor_d, -1, -1)):
            strides[j] = products[i]
        for i_tmp, j in enumerate(range(d-1, minor_d, -1)):
            var i = i_tmp + (d - minor_d)
            strides[j] = products[i]
        # Instantiate the array
        self.data = data.copy()
        self.shape = shape.copy()
        self.strides = strides.copy()

    def write_to(self, mut write: Some[Writer]):
        print(self)