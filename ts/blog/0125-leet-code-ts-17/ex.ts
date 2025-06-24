function merge(unsortedIntervals: number[][]): number[][] {
  if (unsortedIntervals.length === 0) {
    return unsortedIntervals;
  }
  const sorted = unsortedIntervals.sort(
    (a, b) => (a[0] as number) - (b[0] as number),
  );
  const merged: number[][] = [sorted[0] as number[]];

  let lastEnd = (sorted[0] as number[])[1] as number;

  for (let i = 1; i < sorted.length; i++) {
    const interval = sorted[i] as number[];

    // check if current interval starts beore the last one ends. if so, merge
    // it in. otherwise, append it.
    const currentStart = interval[0] as number;
    if (currentStart <= lastEnd) {
      // merge it in
      (merged[merged.length - 1] as number[])[1] = interval[1] as number;
    } else {
      // append
      merged.push(interval);
    }
    lastEnd = (merged[merged.length - 1] as number[])[1] as number;
  }

  return merged;
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
    console.log(`Test ${index + 1}: Input: ${JSON.stringify(test.input)}`);
    console.log(`Expected: ${JSON.stringify(test.expected)}`);
    const result = merge(test.input);
    console.log(`Got: ${JSON.stringify(result)}`);
    console.log(
      `Pass: ${JSON.stringify(result) === JSON.stringify(test.expected)}\n`,
    );
  });
}

// Run the tests
testMergeIntervals();
