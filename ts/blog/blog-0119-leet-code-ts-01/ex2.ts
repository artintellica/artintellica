function twoSumAllPairs(
  nums: number[],
  target: number,
): { numIdx: number; complementIdx: number }[] {
  // mapping complement to index array
  const map: Map<number, number[]> = new Map();
  const results: { numIdx: number; complementIdx: number }[] = [];

  // first, get index for every number
  for (let idx = 0; idx < nums.length; idx++) {
    const num = nums[idx];
    if (!map.has(num)) {
      map.set(num, []);
    }
    (map.get(num) as number[]).push(idx);
  }

  // second pass, find pairs
  for (let idx = 0; idx < nums.length; idx++) {
    const num = nums[idx];
    const complement = target - num;

    // check if complement exists
    if (map.has(complement)) {
      const complementIdxArr = map.get(complement) as number[];
      for (const jdx of complementIdxArr) {
        if (jdx > idx) {
          results.push({
            numIdx: idx,
            complementIdx: jdx,
          });
        }
      }
    }
  }

  return results;
}

console.log(twoSumAllPairs([2, 7, 11, 15], 9)); // solution
console.log(twoSumAllPairs([2, 2, 2, 7, 11, 15], 9)); // three solutions
console.log(twoSumAllPairs([2, 7, 11, 15], 200)); // no solution
