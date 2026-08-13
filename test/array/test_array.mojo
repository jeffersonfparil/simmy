from array import Array
from std.testing import assert_equal, TestSuite

def test_array() raises:
    var arr = Array[DType.float64](
        [1.0, 2.0, 3.0, 4.0],
        [2,2]
    )
    assert_equal(len(arr.data), 4)
    assert_equal(arr.shape, [2, 2])
    
def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()