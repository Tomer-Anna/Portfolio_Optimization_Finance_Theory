from scipy.optimize import minimize
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
import random
from typing import Union, List, Tuple
import logging
# import plotly.io as pio
#
# pio.renderers.default = "svg"
logging.basicConfig(filename='portfolio.log', level=logging.ERROR)


class Portfolio:
    """
    A class for portfolio optimization and analysis.

    Args:
      data (pd.DataFrame): DataFrame with historical asset prices.
      return_calc (str): Accepts either 'log' or 'simple' as the method to compute returns (default is log).
      intervals (int, optional): Number of trading intervals per year (default is 252).
      risk_free_rate (float, optional): Risk-free rate used for calculations (default is 0).
      minimum_weight (float, optional): Minimum weight for assets in the portfolio (default is 0).
      mult (float): multiplier of max sharp return to limit the frontier (default is 1.2).

    Attributes:
      data (pd.DataFrame): Historical asset price data.
      mean_returns (pd.Series): Mean returns for each asset.
      cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
      num_assets (int): Number of assets in the portfolio.
      intervals (int): Number of trading intervals per year.
      risk_free_rate (float): Risk-free rate used for calculations.
      minimum_weight (float): Minimum weight for assets in the portfolio.
    """

    def __init__(self, data, intervals=252, risk_free_rate=0, minimum_weight=0, return_calc='log', mult=1.2):
        self.return_calc = return_calc
        self.data = data
        self.mean_returns, self.cov_matrix = self.initial_process()
        self.num_assets = len(self.mean_returns)
        self.intervals = intervals
        self.risk_free_rate = risk_free_rate
        self.minimum_weight = minimum_weight
        self.mult = mult

    def initial_process(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Perform initial data processing to calculate mean returns and the covariance matrix.

        Returns:
            Tuple[pd.Series, pd.DataFrame]: Mean returns and covariance matrix.
        """
        try:
            if not isinstance(self.data, pd.DataFrame) or self.data.empty:
                raise ValueError("Input data must be a non-empty DataFrame.")

            if self.return_calc == 'simple':
                returns = self.data.pct_change()
            elif self.return_calc == 'log':
                returns = np.log(self.data / self.data.shift(1))

            return returns.mean(), returns.cov()

        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")

    def portfolio_return(self, weights: Union[List[float], np.ndarray]) -> float:
        """
        Calculate the portfolio return.

        Args:
            weights (List[float] or np.ndarray): Portfolio asset weights.

        Returns:
            float: Portfolio return.
        """
        try:
            if not isinstance(weights, (list, np.ndarray)) or len(weights) != self.num_assets:
                raise ValueError(
                    "Portfolio weights must be a list or numpy array of the same length as the number of assets.")
            return np.sum(self.mean_returns * weights) * self.intervals  # transform into annually return

        except Exception as e:
            logging.error(f"Error in portfolio_return: {str(e)}")

    def portfolio_std(self, weights: Union[List[float], np.ndarray]) -> float:
        """
        Calculate the portfolio standard deviation.

        Args:
            weights (List[float] or np.ndarray): Portfolio asset weights.

        Returns:
            float: Portfolio standard deviation.
        """
        try:
            if not isinstance(weights, (list, np.ndarray)) or len(weights) != self.num_assets:
                raise ValueError(
                    "Portfolio weights must be a list or numpy array of the same length as the number of assets.")
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(
                self.intervals)  # transform into annually variance

        except Exception as e:
            logging.error(f"Error in portfolio_std: {str(e)}")

    def get_neg_sharp_ratio(self, weights: Union[List[float], np.ndarray]) -> float:
        """
        Calculate the negative Sharpe ratio of the portfolio.

        Args:
            weights (List[float] or np.ndarray): Portfolio asset weights.

        Returns:
            float: Negative Sharpe ratio.
        """
        p_returns = self.portfolio_return(weights)
        p_std = self.portfolio_std(weights)
        return - (p_returns - self.risk_free_rate) / p_std

    def max_sharp_ratio_portfolio(self) -> minimize:
        """
        Find the portfolio with the maximum Sharpe ratio.

        Returns:
            minimize: Result of the optimization.
        """
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (self.minimum_weight, 1)
        bounds = tuple(bound for asset in range(self.num_assets))
        result = minimize(self.get_neg_sharp_ratio,
                          self.num_assets * [1. / self.num_assets],
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)  # ,options={'ftol': 1e-4, 'eps': 1e-4})

        if not result['success']:
            logging.error(f"Error in max_sharp_ratio_portfolio: {result['message']}")

        return result

    def min_var_portfolio(self, target_return: float = None) -> minimize:
        """
        Find the minimum variance portfolio.

        Args:
            target_return (float, optional): Target portfolio return for a specific portfolio (default is None).

        Returns:
            minimize: Result of the optimization.
        """

        if target_return:
            constraints = (
                {'type': 'eq', 'fun': lambda x: self.portfolio_return(x) - target_return},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            )
        else:
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        bound = (self.minimum_weight, 1)
        bounds = tuple(bound for asset in range(self.num_assets))
        result = minimize(self.portfolio_std,
                          self.num_assets * [1. / self.num_assets],
                          method='SLSQP',
                          bounds=bounds,
                          options={'maxiter': 500},
                          constraints=constraints)

        if not result['success']:
            logging.error(f"Error in min_var_portfolio: {result['message']}")

        return result

    def get_efficient_frontier(self) -> Tuple[
        float, float, pd.DataFrame, float, float, pd.DataFrame, List[float], List[float]]:
        """
        Calculate the efficient frontier and key portfolios.

        Returns:
            Tuple[float, float, pd.DataFrame, float, float, pd.DataFrame, List[float], List[float]:
                - Maximum Sharpe ratio portfolio return, standard deviation, and allocation.
                - Minimum volatility portfolio return, standard deviation, and allocation.
                - List of portfolio standard deviations for the efficient frontier.
                - List of target returns for the efficient frontier.
        """

        # Max Sharpe Ratio Portfolio
        max_sharp_portfolio = self.max_sharp_ratio_portfolio()
        max_sharp_returns = self.portfolio_return(max_sharp_portfolio['x'])
        max_sharp_std = self.portfolio_std(max_sharp_portfolio['x'])
        max_sharp_allocation = pd.DataFrame(max_sharp_portfolio['x'], index=self.mean_returns.index,
                                            columns=['allocation'])

        # Min Volatility Portfolio
        min_vol_portfolio = self.min_var_portfolio()
        min_vol_returns = self.portfolio_return(min_vol_portfolio['x'])
        min_vol_std = self.portfolio_std(min_vol_portfolio['x'])
        min_vol_allocation = pd.DataFrame(min_vol_portfolio['x'], index=self.mean_returns.index, columns=['allocation'])

        # Efficient Frontier
        efficient_frontier_list = []

        target_return_list = np.linspace(min_vol_returns, max_sharp_returns * self.mult, 50)
        for target in target_return_list:
            efficient_frontier_list.append(self.min_var_portfolio(target_return=target)['fun'])

        return max_sharp_returns, max_sharp_std, max_sharp_allocation, min_vol_returns, min_vol_std, min_vol_allocation, efficient_frontier_list, target_return_list

    def simulate_portfolios(self, max_r: float, max_std: float, trials: int = 5000) -> Tuple[
        List[float], List[float], List[List[float]]]:
        """
        Simulate random portfolios with specified constraints.

        Args:
            max_r (float): Maximum allowed return.
            max_std (float): Maximum allowed standard deviation.
            trials (int, optional): Number of portfolio simulations (default is 5000).

        Returns:
            Tuple[List[float], List[float], List[List[float]]:
                - List of portfolio returns.
                - List of portfolio standard deviations.
                - List of portfolio weightings.
        """

        returns, stds, results = [], [], []
        for _ in tqdm(range(trials)):
            # Generate random weights
            weights = np.random.exponential(scale=0.5, size=self.num_assets)

            # Randomly select indices to change to minimum weigh
            indices_to_change = random.sample(range(len(weights)), np.random.randint(1, int(len(weights) * 0.8)))

            # Create a new list with elements changed to minimum weight
            weights = [self.minimum_weight if i in indices_to_change else elem for i, elem in enumerate(weights)]

            # Normalize to ensure the weights sum to 1
            weights /= np.sum(weights)

            # Get retrun and std
            p_return = self.portfolio_return(weights)
            p_std = self.portfolio_std(weights)

            # Make limits of maximum retruns and std
            if p_return > max_r or p_std > max_std:
                pass
            else:
                returns.append(p_return)
                stds.append(p_std)
                results.append(weights)
        return returns, stds, results

    def create_cml(self, max_sharp_return: float, max_sharp_std: float, max_leverage=2.0):
        """
        Create a Capital Market Line (CML) by combining the given maximum Sharpe ratio return and standard deviation.

        Args:
            max_sharp_return (float): The maximum Sharpe ratio return.
            max_sharp_std (float): The standard deviation corresponding to the maximum Sharpe ratio.
            max_leverage (float): The maximum leverage that allowed (default is 200% )
        """
        self.cml = {'returns': [], 'stds': [], 'leverage_returns': [], 'leverage_stds': []}

        for i in np.linspace(0, max_leverage, 60):
            Wp = i
            Wrf = 1 - Wp
            if i <= 1:
                self.cml['returns'].append(Wp * max_sharp_return + Wrf * self.risk_free_rate)
                self.cml['stds'].append(Wp * max_sharp_std)  # + Wrf * risk_free_std <- which always be 0
            else:
                self.cml['leverage_returns'].append(Wp * max_sharp_return + Wrf * self.risk_free_rate)
                self.cml['leverage_stds'].append(Wp * max_sharp_std)  # + Wrf * risk_free_std <- which always be 0

    # Could be outside from the class
    def plot_efficient_frontier(self, with_cml=False) -> go.Figure:
        """
        Plot the efficient frontier and portfolio optimization results.

        Returns:
            go.Figure: Plotly figure displaying the efficient frontier.
        """
        # Get the Efficient Frontier data
        max_sharp_returns, max_sharp_std, _, min_vol_returns, min_vol_std, _, efficient_frontier_list, target_return_list = self.get_efficient_frontier()

        # Get the CML (Capital Market Line) data
        self.create_cml(max_sharp_returns, max_sharp_std)

        # Get the random combination portfolios
        returns, stds, _ = self.simulate_portfolios(max_r=max_sharp_returns * 1.5, max_std=max_sharp_std * 1.5)

        # Efficient Frontier portfolios
        EfficientFrontier = go.Scatter(
            name='Efficient Frontier',
            mode='lines',
            opacity=0.7,
            x=[ef_std for ef_std in efficient_frontier_list],
            y=[target for target in target_return_list],
            line=dict(color='black', width=3, dash='dash')
        )

        # Max Sharp Ratio portfolio
        MaxSharpeRatio = go.Scatter(
            name='Maximium Sharpe Ratio',
            mode='markers',
            x=[max_sharp_std],
            y=[max_sharp_returns],
            marker=dict(color='red', size=14, line=dict(width=1, color='black'))
        )
        # Min volatility portfolio
        MinVol = go.Scatter(
            name='Mininium Volatility',
            mode='markers',
            x=[min_vol_std],
            y=[min_vol_returns],
            marker=dict(color='green', size=14, line=dict(width=1, color='black'))
        )

        # Random Portfolios
        RandomPortfolios = go.Scatter(
            name='Random Weights Portfolios',
            mode='markers',
            opacity=0.5,
            x=stds,
            y=returns,
            line=dict(color='blue', width=0.5)
        )
        # CML (Capital Market Line)
        CML = go.Scatter(
            name='CML (Capital Market Line)',
            mode='lines',
            opacity=0.8,
            x=[cml_std for cml_std in self.cml['stds']],
            y=[cml_return for cml_return in self.cml['returns']],
            line=dict(color='blue', width=2.5)
        )

        # Leveraged CML (Capital Market Line)
        LeveragedCML = go.Scatter(
            name='Leveraged CML (Capital Market Line)',
            mode='lines',
            opacity=0.6,
            x=[cml_std for cml_std in self.cml['leverage_stds']],
            y=[target for target in self.cml['leverage_returns']],
            line=dict(color='blue', width=2.5, dash='dash')
        )
        if with_cml:
            data = [MaxSharpeRatio, MinVol, EfficientFrontier, RandomPortfolios, CML, LeveragedCML]
            title = 'Portfolio Optimization with the Efficient Frontier and CML'

        else:
            data = [MaxSharpeRatio, MinVol, EfficientFrontier, RandomPortfolios]
            title = 'Portfolio Optimization with the Efficient Frontier'

        layout = go.Layout(
            title=title,
            yaxis=dict(title='Annualised Return (%)'),
            xaxis=dict(title='Annualised Volatility (%)'),
            showlegend=True,
            legend=dict(x=0.75, y=0, traceorder='normal',
                        bgcolor='#E2E2E2',
                        bordercolor='black',
                        borderwidth=2),
            width=800,
            height=600)

        fig = go.Figure(data=data, layout=layout)
        return fig.show()
