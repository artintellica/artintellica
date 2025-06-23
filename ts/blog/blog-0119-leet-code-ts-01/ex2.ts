function twoSum(
  nums: number[],
  target: number,
): { numIdx: number; complementIdx: number }[] | null {
  // mapping complement to index
  const map: Map<number, number[]> = new Map();

  for (let idx = 0; idx < nums.length; idx++) {
    const num = nums[idx];
    const complement = target - num;
    const complementIdx = map.get(complement);
    if (complementIdx !== undefined) {
      const numIdx = idx;
      return { numIdx, complementIdx };
    }
    map.set(num, idx);
  }
  return null; // no solution found
}

console.log(twoSum([2, 7, 11, 15], 9)); // solution
console.log(twoSum([2, 7, 11, 15], 200)); // no solution
