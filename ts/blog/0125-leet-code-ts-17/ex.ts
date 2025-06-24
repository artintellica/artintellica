function merge(intervals: number[][]): number[][] {
  const sorted = intervals.sort((a, b) => (a[0] as number) - (b[0] as number));

  console.log(sorted)

  return [[]];
}

function testMergeIntervals(): void {
  const testCases = [
    {
      input: [
        [1, 3],
        [2, 6],
        [8, 10],
        [15, 18],
      ],
      expected: [
        [1, 6],
        [8, 10],
        [15, 18],
      ],
    },
    {
      input: [
        [1, 4],
        [4, 5],
      ],
      expected: [[1, 5]],
    },
    {
      input: [
        [1, 4],
        [0, 4],
      ],
      expected: [[0, 4]],
    },
    {
      input: [
        [1, 4],
        [2, 3],
      ],
      expected: [[1, 4]],
    },
    {
      input: [
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [1, 10],
      ],
      expected: [[1, 10]],
    },
    { input: [], expected: [] },
  ];

  testCases.forEach((test, index) => {
    const result = merge(test.input);
    console.log(`Test ${index + 1}: Input: ${JSON.stringify(test.input)}`);
    console.log(`Expected: ${JSON.stringify(test.expected)}`);
    console.log(`Got: ${JSON.stringify(result)}`);
    console.log(
      `Pass: ${JSON.stringify(result) === JSON.stringify(test.expected)}\n`,
    );
  });
}

// Run the tests
testMergeIntervals();
