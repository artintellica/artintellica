function maxProfitII(prices: number[]): number {
  // If the array is empty or has only one element, no profit is possible
  if (prices.length < 2) {
    return 0;
  }

  // Initialize total profit
  let totalProfit = 0;

  // Iterate through the array
  for (let i = 1; i < prices.length; i++) {
    // If current price is higher than previous day's price,
    // we can make a profit by buying yesterday and selling today
    if ((prices[i] as number) > (prices[i - 1] as number)) {
      totalProfit += (prices[i] as number) - (prices[i - 1] as number);
    }
  }

  return totalProfit;
}

console.log(maxProfitII([7, 1, 5, 3, 6, 4])); // Output: 7
console.log(maxProfitII([1, 2, 3, 4, 5])); // Output: 4 (profit on each consecutive day)
console.log(maxProfitII([7, 6, 4, 3, 1])); // Output: 0 (no profit possible)
