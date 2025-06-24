function merge(intervals: number[][]): number[][] {
  // If the input is empty, return an empty array
  if (intervals.length === 0) {
    return [];
  }

  // Sort intervals based on start time
  intervals.sort((a, b) => (a[0] as number) - (b[0] as number));

  // Initialize the result array with the first interval
  const merged: number[][] = [intervals[0] as number[]];

  // Iterate through the sorted intervals starting from the second one
  for (let i = 1; i < intervals.length; i++) {
    const currentInterval = intervals[i] as number[];
    const lastMerged = merged[merged.length - 1] as number[];

    // Check if the current interval overlaps with the last merged interval
    if ((currentInterval[0] as number) <= (lastMerged[1] as number)) {
      // If overlap, update the end time of the last merged interval
      lastMerged[1] = Math.max(
        lastMerged[1] as number,
        currentInterval[1] as number,
      );
    } else {
      // If no overlap, add the current interval to the result
      merged.push(currentInterval);
    }
  }

  return merged;
}

// Test cases
console.log(
  merge([
    [1, 3],
    [2, 6],
    [8, 10],
    [15, 18],
  ]),
); // Output: [[1,6],[8,10],[15,18]]
console.log(
  merge([
    [1, 4],
    [4, 5],
  ]),
); // Output: [[1,5]]
console.log(
  merge([
    [1, 4],
    [0, 4],
  ]),
); // Output: [[0,4]]
console.log(
  merge([
    [1, 4],
    [2, 3],
  ]),
); // Output: [[1,4]]
