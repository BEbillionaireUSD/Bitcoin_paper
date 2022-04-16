# -*- coding: UTF-8 -*-
import os
import numpy as np
import pandas as pd
import quantstats as qs
import xlwt

from os import path
from typing import Dict

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.policies import BasePolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

from lib.env.TradingEnv import TradingEnv
from lib.env.reward import BaseRewardStrategy, IncrementalProfit, WeightedUnrealizedProfit
from lib.data.providers.dates import ProviderDateFormat
from lib.data.providers import BaseDataProvider,  StaticDataProvider, ExchangeDataProvider
from lib.util.logger import init_logger


def make_env(data_provider: BaseDataProvider, rank: int = 0, seed: int = 0):#构建环境
    def _init():
        env = TradingEnv(data_provider)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)

    return _init


class RLTrader:#构建PPO模型
    data_provider = None
    study_name = 'PPO2__WeightedUnrealizedProfit'

    def __init__(self,
                 model: BaseRLModel = PPO2,
                 policy: BasePolicy = MlpLnLstmPolicy,
                 reward_strategy: BaseRewardStrategy = WeightedUnrealizedProfit,#设定奖励函数
                 exchange_args: Dict = {},
                 **kwargs):
        self.logger = kwargs.get('logger', init_logger(__name__, show_debug=kwargs.get('show_debug', True)))

        self.Model = model
        self.Policy = policy
        self.Reward_Strategy = reward_strategy
        self.exchange_args = exchange_args
        self.tensorboard_path = kwargs.get('tensorboard_path', None)
        self.input_data_path = kwargs.get('input_data_path', 'data/input/coinbase-1h-btc-usd.csv')#这里的data可选择1h/1d

        self.date_format = kwargs.get('date_format', ProviderDateFormat.DATETIME_HOUR_24)

        self.model_verbose = kwargs.get('model_verbose', 1)#show an animated progress bar
        self.n_envs = kwargs.get('n_envs', os.cpu_count())#使用cpu
        self.n_minibatches = kwargs.get('n_minibatches', self.n_envs)
        self.train_split_percentage = kwargs.get('train_split_percentage', 0.8)#80%训练集+20%测试集
        self.data_provider = kwargs.get('data_provider', 'static')

        self.initialize_data()

        self.logger.debug(f'Initialize RLTrader: {self.study_name}')

    def initialize_data(self):#初始化数据
        if self.data_provider == 'static':
            if not os.path.isfile(self.input_data_path):
                class_dir = os.path.dirname(__file__)
                self.input_data_path = os.path.realpath(os.path.join(class_dir, "../{}".format(self.input_data_path)))

            data_columns = {'Date': 'Date', 'Open': 'Open', 'High': 'High',
                            'Low': 'Low', 'Close': 'Close', 'Volume': 'VolumeFrom'}

            self.data_provider = StaticDataProvider(date_format=self.date_format,
                                                    csv_data_path=self.input_data_path,
                                                    data_columns=data_columns)
        elif self.data_provider == 'exchange':
            self.data_provider = ExchangeDataProvider(**self.exchange_args)

        self.logger.debug(f'Initialized Features: {self.data_provider.columns}')
    #训练
    def train(self,
              n_epochs: int = 30, #训练30轮
              save_every: int = 1,#每个样本保存一次
              test_trained_model: bool = True,
              render_test_env: bool = True,
              render_report: bool = True,
              save_report: bool = True):
        train_provider, test_provider = self.data_provider.split_data_train_test(self.train_split_percentage)

        del test_provider

        train_env = SubprocVecEnv([make_env(train_provider, i) for i in range(self.n_envs)])


        model = self.Model(self.Policy,
                           train_env,
                           verbose=self.model_verbose,
                           nminibatches=self.n_minibatches,
                           tensorboard_log=self.tensorboard_path)

        self.logger.info(f'Training for {n_epochs} epochs')

        steps_per_epoch = len(train_provider.data_frame)

        for model_epoch in range(0, n_epochs):
            self.logger.info(f'[{model_epoch}] Training for: {steps_per_epoch} time steps')

            model.learn(total_timesteps=steps_per_epoch)

            if model_epoch % save_every == 0:
                model_path = path.join('data', 'agents', f'{self.study_name}__{model_epoch}.pkl')
                model.save(model_path)

                if test_trained_model:
                    self.test(model_epoch,
                              render_env=render_test_env,
                              render_report=render_report,
                              save_report=save_report)

        self.logger.info(f'Trained {n_epochs} models')
    #测试
    def test(self, model_epoch: int = 0, render_env: bool = True, render_report: bool = True, save_report: bool = True):
        train_provider, test_provider =  self.data_provider.split_data_train_test(self.train_split_percentage)

        del train_provider

        init_envs = DummyVecEnv([make_env(test_provider) for _ in range(self.n_envs)])

        model_path = path.join('data', 'agents', f'{self.study_name}__{model_epoch}.pkl')
       
        model = self.Model.load(model_path, env=init_envs)

        test_env = DummyVecEnv([make_env(test_provider) for _ in range(1)])

        self.logger.info(f'Testing model ({self.study_name}__{model_epoch})')

        zero_completed_obs = np.zeros((self.n_envs,) + init_envs.observation_space.shape)
        zero_completed_obs[0, :] = test_env.reset()

        state = None
        rewards = []
        
        net_worths_log = []
        
        for _ in range(len(test_provider.data_frame)):
            action, state = model.predict(zero_completed_obs, state=state)
            obs, reward, done, info = test_env.step([action[0]])

            zero_completed_obs[0, :] = obs

            rewards.append(reward)
            

            if render_env:
                net = test_env.render(mode='system')
                net_worths_log.append(net)

            if done:
                net_worths = pd.DataFrame({
                    'Date': info[0]['timestamps'],
                    'Balance': info[0]['net_worths'],
                })
                excel_path = path.join(f'{self.study_name}__{model_epoch}.xls')
                excel_path = './data/test/'+excel_path
                net_worths.to_excel(excel_path)

                net_worths.set_index('Date', drop=True, inplace=True)
                returns = net_worths.pct_change()[1:]

                if render_report:
                    qs.plots.snapshot(returns.Balance, title='RL Trader Performance')

                if save_report:
                    reports_path = path.join('data', 'reports', f'{self.study_name}__{model_epoch}.html')
                    qs.reports.html(returns.Balance, file=reports_path)
        file_path = path.join(f'{self.study_name}__{model_epoch}.txt')
        file = open('./data/test/'+file_path,'w+')
        for net in net_worths_log:
            file.write(str(net)+'\n')
        self.logger.info(
            f'Finished testing model ({self.study_name}__{model_epoch}): ${"{:.2f}".format(np.sum(rewards))}')
