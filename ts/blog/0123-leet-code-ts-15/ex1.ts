function maxProfit(prices: number[]): number {
  let minPrice: number = prices[0] as number;
  let profit = 0;

  for (const price of prices) {
    if (price < minPrice) {
      minPrice = price;
    }
    if (price - minPrice > profit) {
      profit = price - minPrice;
    }
  }
  return profit;
}

console.log(maxProfit([7, 1, 5, 3, 6, 4])); // Output: 5 (buy at 1, sell at 6)
console.log(maxProfit([7, 6, 4, 3, 1])); // Output: 0 (no profit possible)
console.log(maxProfit([1, 2])); // Output: 1 (buy at 1, sell at 2)
console.log(maxProfit([2])); // Output: 0 (can't sell)
