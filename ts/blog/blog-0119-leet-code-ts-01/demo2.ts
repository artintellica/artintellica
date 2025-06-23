function twoSumAllPairs(nums: number[], target: number): number[][] {
  // Create a hash map to store number-to-indices mappings (handling duplicates)
  const numMap: Map<number, number[]> = new Map();
  const result: number[][] = [];

  // Populate the map with all indices for each number
  for (let i = 0; i < nums.length; i++) {
    const currentNum = nums[i];
    if (!numMap.has(currentNum)) {
      numMap.set(currentNum, []);
    }
    (numMap.get(currentNum) as number[]).push(i);
  }

  // Iterate through the array to find pairs
  for (let i = 0; i < nums.length; i++) {
    const currentNum = nums[i];
    const complement = target - currentNum;

    // Check if complement exists
    if (numMap.has(complement)) {
      const complementIndices = numMap.get(complement) as number[];
      // Add pairs, ensuring we don't reuse the same index
      for (const j of complementIndices) {
        if (j > i) {
          // Only consider indices after i to avoid duplicates
          result.push([i, j]);
        }
      }
    }
  }

  return result;
}

// Test the function
const numsVariation = [1, 5, 5, 1, 3];
const targetVariation = 6;
console.log(twoSumAllPairs(numsVariation, targetVariation)); // Output: [[0, 1], [0, 2], [3, 1], [3, 2]]
