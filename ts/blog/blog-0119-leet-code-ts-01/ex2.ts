function twoSumAllPairs(
  nums: number[],
  target: number,
): { numIdx: number; complementIdx: number }[] {
  // mapping complement to index array
  const map: Map<number, number[]> = new Map();
  const results: { numIdx: number; complementIdx: number }[] = [];

  // first pass - compute complements
  for (let idx = 0; idx < nums.length; idx++) {
    const num = nums[idx];
    const complement = target - num;
    let complementIdxArr = map.get(complement);
    if (complementIdxArr !== undefined) {
      complementIdxArr.push(idx);
    } else {
      complementIdxArr = []
    }
    map.set(complement, complementIdxArr)
  }

  // second pass - accumulate complements
  for (let idx = 0; idx < nums.length; idx++) {
    const num = nums[idx];
    const complement = target - num;
    const complementIdxArr = map.get(complement);
    if (complementIdxArr !== undefined) {
      for (const complementIdx of complementIdxArr) {
        results.push({
          numIdx: idx,
          complementIdx,
        });
      }
    }
  }

  return results;
}

console.log(twoSumAllPairs([2, 7, 11, 15], 9)); // solution
console.log(twoSumAllPairs([2, 7, 11, 15], 200)); // no solution
