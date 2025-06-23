function twoSum(nums: number[], target: number): number[] {
  // Create a hash map to store number-to-index mappings
  const numMap: Map<number, number> = new Map();

  // Iterate through the array
  for (let i = 0; i < nums.length; i++) {
    const currentNum = nums[i];
    // Calculate the complement needed to reach the target
    const complement = target - currentNum;

    // Check if complement exists in the map
    if (numMap.has(complement)) {
      // Return the indices of the two numbers
      return [numMap.get(complement) as number, i];
    }

    // If not found, add the current number and its index to the map
    numMap.set(currentNum, i);
  }

  // No solution found (though problem guarantees one exists)
  return [];
}

// Test the function
const nums = [2, 7, 11, 15];
const target = 9;
console.log(twoSum(nums, target)); // Output: [0, 1]
