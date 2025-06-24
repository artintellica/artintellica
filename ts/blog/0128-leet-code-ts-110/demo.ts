function findMedianSortedArrays(nums1: number[], nums2: number[]): number {
  // Ensure nums1 is the shorter array for efficiency in binary search
  if (nums1.length > nums2.length) {
    [nums1, nums2] = [nums2, nums1];
  }

  const m: number = nums1.length;
  const n: number = nums2.length;
  let left = 0;
  let right: number = m;

  while (left <= right) {
    // Partition points for nums1 and nums2
    const x: number = Math.floor((left + right) / 2);
    const y: number = Math.floor((m + n + 1) / 2) - x;

    // Get left and right values for nums1
    const left1: number =
      x === 0 ? Number.NEGATIVE_INFINITY : (nums1[x - 1] as number);
    const right1: number =
      x === m ? Number.POSITIVE_INFINITY : (nums1[x] as number);

    // Get left and right values for nums2
    const left2: number =
      y === 0 ? Number.NEGATIVE_INFINITY : (nums2[y - 1] as number);
    const right2: number =
      y === n ? Number.POSITIVE_INFINITY : (nums2[y] as number);

    // Check if partition is valid
    if (left1 <= right2 && left2 <= right1) {
      // If total length is odd, median is the max of left parts
      if ((m + n) % 2 === 1) {
        return Math.max(left1, left2);
      }
      // If total length is even, median is average of max(left) and min(right)
      return (Math.max(left1, left2) + Math.min(right1, right2)) / 2;
    }
    if (left1 > right2) {
      // If left1 is too big, move partition left in nums1
      right = x - 1;
    } else {
      // If left2 is too big, move partition right in nums1
      left = x + 1;
    }
  }

  throw new Error("Input arrays are not sorted");
}

{
  const nums1: number[] = [1, 3];
  const nums2: number[] = [2];
  console.log(findMedianSortedArrays(nums1, nums2)); // Output: 2
}
{
  const nums1: number[] = [1, 2];
  const nums2: number[] = [3, 4];
  console.log(findMedianSortedArrays(nums1, nums2)); // Output: 2.5
}
{
  const nums1: number[] = [0, 0];
  const nums2: number[] = [0, 0];
  console.log(findMedianSortedArrays(nums1, nums2)); // Output: 0
}
{
  const nums1: number[] = [1, 3, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7];
  const nums2: number[] = [0, 1, 2, 3, 4, 5, 56, 57, 58, 59, 60, 61, 62, 63];
  console.log(findMedianSortedArrays(nums1, nums2));
}
