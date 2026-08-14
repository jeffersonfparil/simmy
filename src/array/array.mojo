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

    def __init__(
        out self,
        data: List[Scalar[Self.dtype]],
        shape: List[Int],
        *,
        minor_dimension: Int = -1,
    ) raises:
        var n: Int = len(data)
        var m: Int = 1
        for s in shape:
            m *= s
            if m == 0:
                break
        if n != m:
            raise Error(
                "Invalid shape ("
                + String(shape)
                + ") for data of length "
                + String(n)
                + "!"
            )
        for s in shape:
            if s < 1:
                raise Error(
                    "Dimension size must be greater than zero, got "
                    + String(s)
                    + "!"
                )
        var d: Int = len(shape)
        if (minor_dimension > d - 1) or (minor_dimension < -d):
            raise Error(
                "Invalid minor dimension for a "
                + String(d)
                + "-dimensional array!"
            )
        var minor_d = (
            d + minor_dimension if minor_dimension < 0 else minor_dimension
        )
        # To find the strides given the minor dimenion:
        # - First, we find the products of the dimensions sizes starting from the minor dimension which starts at 1 always
        var products = List[Int](length=d, fill=1)
        var counter: Int = (
            1  # We skip the first one because we always start at 1
        )
        for i in range(minor_d, 0, -1):
            products[counter] = products[counter - 1] * shape[i]
            counter += 1
        for i in range(d - 1, minor_d, -1):
            products[counter] = products[counter - 1] * shape[i]
            counter += 1
        # - Finally, we place the dimension size products to their corresponding stride locations
        var strides = List[Int](length=d, fill=1)
        counter = 0
        for i in range(minor_d, -1, -1):
            strides[i] = products[counter]
            counter += 1
        for i in range(d - 1, minor_d, -1):
            strides[i] = products[counter]
            counter += 1
        # Instantiate the array
        self.data = data.copy()
        self.shape = shape.copy()
        self.strides = strides.copy()

    def write_to(self, mut writer: Some[Writer]):
        writer.write(len(self.shape), "-D Array\n")
        writer.write("  - shape: ", self.shape, "\n")
        writer.write("  - size: ", len(self.data), "\n")
        writer.write("  - strides: ", self.strides, "\n")
        writer.write("  - dtype: ", self.dtype, "\n")
        var MAX_ROWS: Int = 7
        var MAX_COLS: Int = 7
        var is_1d: Bool = True if len(self.shape) == 1 else False
        var all_rows: Bool = True if self.shape[0] <= MAX_ROWS else False
        var all_cols: Bool = (
            True if not is_1d and self.shape[1] <= MAX_COLS else False
        )
        if is_1d:
            var n = len(self.data)
            var indices: List[Int] = List(range(3+1))
            indices.extend(List(range(n - 3, n)))
            for i in indices:
                var x = self.data[i]
                if i == 0:
                    writer.write("⎡ ", x, " ⎤\n")
                elif i < 3 or (i > 3 and i < n - 1):
                    writer.write("⎢ ", x, " ⎥\n")
                elif not all_rows and i == 3:
                    writer.write("⎢ ... ⎥\n")
                else:
                    writer.write("⎣ ", x, " ⎦\n")
        if not is_1d:
            var n: Int = self.shape[0]
            var p: Int = self.shape[1]
            var i_indices: List[Int] = List(range(3+1))
            var j_indices: List[Int] = List(range(3+1))
            if all_rows:
                i_indices = List(range(n))
            else:
                i_indices.extend(List(range(n-3, n)))
            if all_cols:
                j_indices = List(range(p))
            else:
                j_indices.extend(List(range(p-3, p)))
            var ds: List[Int] = [0] if len(self.shape) < 3 else List(range(len(self.shape) - 2))
            for d in ds:
                var z: Int = 0 if len(self.shape) < 3 else d + 1
                writer.write("\n")
                if len(self.shape) > 2:
                    var x = ", ".join(["0" for _ in range(z)])
                    var y = ", ".join(["0" for _ in range(len(self.shape)-(z+2))])
                    writer.write("(0:" + String(self.shape[0]) + ", 0:" + String(self.shape[1]) + x + String(", ", d) + ", " + y + ")\n")
                for i in i_indices:
                    if i == 0:
                        writer.write("⎡ ")
                    elif i < n - 1:
                        writer.write("⎢ ")
                    else:
                        writer.write("⎣ ")
                    for j in j_indices:
                        if (not all_rows and i == 3) or (not all_cols and j == 3):
                            writer.write("... ")
                        else:
                            var k: Int = i * self.strides[0] + j * self.strides[1] + z
                            writer.write(self.data[k], " ")
                    if i == 0:
                        writer.write("⎤\n")
                    elif i < n - 1:
                        writer.write("⎥\n")
                    else:
                        writer.write("⎦\n")
            