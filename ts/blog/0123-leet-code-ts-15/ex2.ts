function maxProfit2(prices: number[]): number {
  if (prices.length < 2) {
    return 0;
  }

  let profit = 0;

  for (let i = 1; i < prices.length; i++) {
    const price = prices[i] as number;
    const yesterdayPrice = prices[i - 1] as number;
    if (price > yesterdayPrice) {
      profit += price - yesterdayPrice;
    }
  }

  return profit;
}

console.log(maxProfit2([7, 1, 5, 3, 6, 4])); // Output: 7
console.log(maxProfit2([1, 2, 3, 4, 5])); // Output: 4 (profit on each consecutive day)
console.log(maxProfit2([7, 6, 4, 3, 1])); // Output: 0 (no profit possible)
