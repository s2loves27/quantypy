#os - 폴더생성, 파일 경로 준비
#logging 학습과정중에 정보를 기록

#abc 추상 클래스를 정의
#abstract base class의 약자로 추상 클래스를 정의.
#@abstractmethod 데코레이터를 사용해 추상 메서드를 선언 할 수 있다.

#time 학습 시간을 측정
#numpy 배열 자료 구조 조작
#util sigmoid - 정책 신경망 학습 레이블을 생성
import os
import logging
import abc
import collections
import threading
import time
import numpy as np
from utils import sigmoid
from environment import Environment
from agent import Agent
from networks import Network, DNN, LSTMNetwork, CNN
from visualizer import Visualizer

## DQNLearner, PolicyGradientLearner, ActorCriticLearner, A2CLearner등을 상속하는 상위 클래스

class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()
    """
    인자 확인
    rl_method - 학습 기법 ("dqn, pg, ac, a2c, a3c 등)
    stock_code - 주식 종목 코드
    chart_data - 주식 일봉 차트 데이터
    training_data - 학습을 위한 전처리 된 학습 데이터
    min_trading_unit - 투자 최소 단위
    max_trading_unit - 투자 최대 단위
    (주식 종목 마다 주가의 스케일이 달라 투자 주식수 단위가 다르기 때문에 적절한 투자 단위 설정이 중요
    delayed_reward_threshold - 지연 보상 임곗값 ( 수익률 이나 손실률이 이 임계 값 보다 클 경우 지연 보상이 발생)
    mini_batch_size - 미니 배치 학습을 위한 크기 - 데이터가 쌓이는 데도 포트폴리오 가치가 크게 변하지 않아
    지연 보상이 발생하지 않는 경우. 바로 학습을 시킬 수 있다.
    net - 정책 신경망으로 사용할 신경망 클래스
    n_step - LSTM, CNN 신경망에서 사용하는 샘플 묶음의 크기
    lr - 학습 속도(너무 크면 학습이 제대로 진행 되지 않음
    value_network - 가치 신경망
    policy_network - 정책 진명망
    output_path - 학습 결과가 저장될 경로.
    """
    def __init__(self, rl_method='rl', stock_code=None, 
                chart_data=None, training_data=None,
                min_trading_unit=1, max_trading_unit=2, 
                delayed_reward_threshold=.05,
                net='dnn', num_steps=1, lr=0.001,
                value_network=None, policy_network=None,
                output_path='', reuse_models=True):
        # 인자 확인
        # assert <condition>
        # condition이 만족하지 않으면 assertionError 예외 발생.
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps > 0
        assert lr > 0

        # 강화학습 기법 설정
        # 'dqn', 'pg', 'ac', 'a2c', 'a3c'
        self.rl_method = rl_method
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        # 에이전트를 변경 함으로써 새로운 환경을 만들어 줄 수 있다.
        self.agent = Agent(self.environment,
                    min_trading_unit=min_trading_unit,
                    max_trading_unit=max_trading_unit,
                    delayed_reward_threshold=delayed_reward_threshold)
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        # network 관련 설정
        # net : 'dnn', 'lstm', 'cnn'
        # num_steps : 몇번 돌것인가?
        # lr : 학습률
        # value_network : ....
        # policy_network : ...
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리
        # 강화학습 과정에서 발생하는 각종 데이터를 쌓아두기 위해 momory_*라는 이름의 변수를 사용.
        # [학습데이터, 수행한 행동, 확득 보상, 행동의 예측 가치, 행동의 예측 확률, 포트 폴리오 가치,
        # 주식 보유수, 탐험위치, 학습 위치등]
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0
        # 로그 등 출력 경로
        self.output_path = output_path

    # 가치 신경망 생성 함수.
    """
    net이 'dnn' 이면 DNN 클래스로 가치 신경망을 생성하고,
    'lstm' 이면 LSTMNetwork 클래스를, 'cnn' 이면 CNN
    클래스로 가치 신경망을 생성합니다. 이 클래스들은
    Network 클래스를 상송하므로 Network 클래스의 함수를 모두 가지고 있습니다.
    """

    def init_value_network(self, shared_network=None, 
            activation='linear', loss='mse'):
        if self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, shared_network=shared_network, 
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network, 
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.value_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network, 
                activation=activation, loss=loss)
        if self.reuse_models and \
            os.path.exists(self.value_network_path):
                self.value_network.load_model(
                    model_path=self.value_network_path)

    """
    가치 신경망을 생성하는 init_value_network() 함수와 매우 유사한 함수 입니다.
    차이는 활성화 함수 activation 인자로 
    가치 신경망은 'linear'
    정책 신경망은 'sigmoid'를 쓴다는 점입니다
    
    정책 신경망은 샘플에 대해서 PV를 높이기 위해 취하기 좋은 행동에 대한 분류 모델.
    활성화 함수로 시그모이드를 써서 결과값이 0과 1사이로 나오게 해서 확률로 사용할 수 있게 했습니다. 
    linear, sigmoid 모두 -- 활성화 함수 ( 0 과 ,1 를 만들기 위한것)
    """
    def init_policy_network(self, shared_network=None, 
            activation='sigmoid', loss='binary_crossentropy'):
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, shared_network=shared_network, 
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network, 
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.policy_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network, 
                activation=activation, loss=loss)
        if self.reuse_models and \
            os.path.exists(self.policy_network_path):
            self.policy_network.load_model(
                model_path=self.policy_network_path)

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

    """
    환경 객체의 observe() 함수 - 차트 데이터의 현재 인덱스에서 다음 인덱스 데이터를 읽게한다.
    if(학습데이터에 다음 인덱스 데이터가 존재하면 Training_data_idx 변수를 1만큼 증가.
    training_data 배열에서 train_data_idx 인덱스의 데이터를 받아와서 sample에 저장.
    다음으로 sample에 에이전트 상태를 추가해 sample에 28값으로 저장.
    
    각종 Trainingdata + 주식 보유 비율 + 포트폴리오 가치 비율 
    """
    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[
                self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    #신경망을 학습하기 위해 [배치 학습데이터를 생성하는곳
    #추상 메서드로서 ReinforcementLearner 클래스의 하위 클래스들은 반드시 이 함수를 구현 해야한다.
    #만약 구현하지 않으면 NotImplemented 예외 발생.
    @abc.abstractmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        pass

    #신경망을 학습 시키는 곳.
    """
     get_batch() 함수를 호출해 배치 학습 데이터를 생성하고 가치 신경망과 정책 신경망을
     학습하기 위해 신경망 클래스 train_on_batch() 함수를 호출한다.
     가치 신경망은 DQNLearner, ActorCriticLeaner, A2CLearner에서 학습
     정책 신경망은 PolicyGradientLearner, ActorCriticLearner, A2Clearner에서 학습.
     학습 후 발생하는 손실(loss)를 반환합니다. 가치 신경망과 정책 신경망을 모두 학습하는 경우
     두 학습 손실을 합산해 반환합니다.
    """
    def update_networks(self, 
            batch_size, delayed_reward, discount_factor):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch(
            batch_size, delayed_reward, discount_factor)
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy)
            return loss
        return None

    """
        배치 학습 데이터의 크기를 정하고, update_network()함수를 호출합니다.
        그리고 반환 받은 학습 손실 값인 _loss를 loss에 더합니다.
        loss는 에포크 동안의 총 합습 손실을 가지게 됩니다.
        learning_cnt에 학습 횟수를 저장하고, 
        나중에 loss를 learning_cnt로 나누어 에포크의 학습 손실로 여깁니다.
        그리고, memory_learning_idx에 학습 위치를 저장합니다. 
        Full - 에포크 동안 쌓은 모든 데이터를 학습에 사용할지 말지를 full 인자로
        받으며, 
        full = True - 전체 데이터에 대해 학습을 수행 (에포크 종료후 가치 신경망 추가 학습 용도)
        full = False - 이전 학습 종료후 학습 시작점 부터 지금 까지.
    """
    def fit(self, delayed_reward, discount_factor, full=False):
        batch_size = len(self.memory_reward) if full \
            else self.batch_size
        # 배치 학습 데이터 생성 및 신경망 갱신
        if batch_size > 0:
            _loss = self.update_networks(
                batch_size, delayed_reward, discount_factor)
            if _loss is not None:
                #abs 절대값
                #학습 손실 = loss / learning_cnt
                self.loss += abs(_loss)
                self.learning_cnt += 1
                #학습 위치 저장.
                self.memory_learning_idx.append(self.training_data_idx)
            self.batch_size = 0

    """
        가시화함수 - 에이전트 행동(momory_action), 보유 주식 수(memory_num_stocks), 
        가치 신경망 출력(momory_value), 정책 신경망 출력(momory_policy)
        포트폴리오 가치(momory_pv), 탐험 위치(memory_exp_idx), 학습 위치(memory_learning_idx)
    """
    """
        LSTM 신경망과 CNN 신경망을 사용하는 경우 에이전트 행동, 보유 주식 수, 가치 신경망 출력
        정책 신경망 출력, 포트폴리오 가치는 환경의 일봉 수 보다 num_step -1 만큼 부족하기 때문에
        num_steps -1 만큼 의미 없는 값을 첫 부분에 채워 준다.
        *** 왜냐면 num_step 개씩 들어가기 때문에 ***
    """

    """
        *** 파이썬 팁 ***
        파이썬에서는 리스트에 곱하기를 하면 똑같은 리스트를 뒤에 붙여 줍니다
        ex, [1, 2, 3] * 3 = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    """
    """
        객체의 visualizer의 plot() 함수를 호출.
        그리고 생성된 에포크 결과 그림을 PNG 그림 파일로 저장.
    """
    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] \
            * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) \
            + self.memory_num_stocks
        if self.value_network is not None:
            self.memory_value = [np.array([np.nan] \
                * len(Agent.ACTIONS))] * (self.num_steps - 1) \
                    + self.memory_value
        if self.policy_network is not None:
            self.memory_policy = [np.array([np.nan] \
                * len(Agent.ACTIONS))] * (self.num_steps - 1) \
                    + self.memory_policy
        self.memory_pv = [self.agent.initial_balance] \
            * (self.num_steps - 1) + self.memory_pv
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, 
            epsilon=epsilon, action_list=Agent.ACTIONS, 
            actions=self.memory_action, 
            num_stocks=self.memory_num_stocks, 
            outvals_value=self.memory_value, 
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx, 
            learning_idxes=self.memory_learning_idx,
            initial_balance=self.agent.initial_balance, 
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(
            self.epoch_summary_dir, 
            'epoch_summary_{}.png'.format(epoch_str))
        )

    """
        ReinforcementLearner 클래스의 핵심 함수!! - 무조건 이해!
        num_epoches - 총 수행할 반복 학습 횟수
        반복학습을 거치면서 가치 신경망과 정책 신경망이 점점 포트폴리오
        가치를 높이는 방향으로 갱신되기 때문에 충분한 반복 횟수를 정해 줘야한다.
        그러나 num_epoches를 너무 크게 잡으면 학습에 소요되는 시간이 너무 길어 지므로
        적절하게 정해야 한다. 
        
        balance - 에이전트 초기 투자 자본금을 정하기 위한 인자
        
        discount_factor - 상태-행동 가치를 구할 때 적용할 할인률.
        보상이 발생했을 때 그 이전 보상이 발생한 시점과 현재 보상이 발생한 시점 사이에서
        수행한 행동 전체에 현재의 보상이 영향을 미친다. 이때 과거로 갈수록 현재 보상을
        적용할 판단 근거가 흐려지기 때문에 먼 과거의 행동일수록 현재의 보상을 약하게 적용합니다. 
        (현재 보상이 발생 한 경우. 과거의 행동일 수록 영향이 적고, 현재에 가까울 수록 영향이 크다.)
        
        learning - 학습 유무를 정하는 boolean 값.
        학습을 마치면 학습된 가치 신경망 모델과 정책 신경망 모델이 만들어 진다.
        이렇게 학습을 해서 신경망 모델을 만들고자 한다면, learning을 True로 학습된 모델을 
        가지고 투자 시뮬레이션만 하려 한다면 learning을 False로 주면 된다.
    """

    def run(
        self, num_epoches=100, balance=10000000,
        discount_factor=0.9, start_epsilon=0.5, learning=True):
        info = "[{code}] RL:{rl} Net:{net} LR:{lr} " \
            "DF:{discount_factor} TU:[{min_trading_unit}," \
            "{max_trading_unit}] DRT:{delayed_reward_threshold}".format(
            code=self.stock_code, rl=self.rl_method, net=self.net,
            lr=self.lr, discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit, 
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        )
        with self.lock:
            logging.info(info)

        # 시작 시간
        time_start = time.time()

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        self.epoch_summary_dir = os.path.join(
            self.output_path, 'epoch_summary_{}'.format(
                self.stock_code))
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        # 학습한 epoches 중 가장 높은 포트폴리오 가치
        # 수행한 에포크중 수익이 발생한 에포크수(초기 자본금 보다 PV가 높아진 에포크 수)
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        """
            time_start_epoch - epoch 현재 시간 저장
            (한 에포크를 수행하는데 걸린시간 예측을 위해서)
            
            q_sample - num_step만큼 샘플을 담아둘 Deque - 양방향 큐
        
            epsilon - 탐험률(무작위 투자 비율) 
            ex , 1 - 100% 무작위 투자   
        """
        for epoch in range(num_epoches):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps)
            
            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            # reset_exploration를 왜하는 것인가?
            if learning:
                epsilon = start_epsilon \
                    * (1. - float(epoch) / (num_epoches - 1))
                self.agent.reset_exploration()
            else:
                epsilon = start_epsilon

            # 하나의 Epoch를 수행하는 while문.
            while True:
                # 샘플 생성
                # build_sample 함수를 호출해 환경 객체로 부터 하나의 샘플을 읽어 옵니다.
                # next_sample이 None이라면 마지막까지 데이터를 읽은 것이므로 while 반복문 정료.
                next_sample = self.build_sample()
                if next_sample is None:
                    break

                # num_steps만큼 샘플 저장
                # lstm, cnn인 경우 num_steps 개수 만큼 샘플이 준비 돼야 행동을 결정 할 수 있기 때문에
                # 샘플 큐에 샘플이 모두 찰때 까지 continue를 통해 이후 로직을 건너 뜁니다.
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                # 각 신경망(value_network, policy_network) 객체의
                # predict() 함수를 호출해 예측 행동 가치와 예측 행동 확률을 구한다.
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(
                        list(q_sample))
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(
                        list(q_sample))
                
                # 신경망 또는 탐험에 의한 행동 결정
                # 위에서 구한 예측 행동 가치와 예측 행동 확률을 통해
                # 투자 행동을 결정한다.
                # epsilon 값에 따라 무작위 or 신경망 출력
                # action - 결정한 행동
                # confidence - 결정에 대한 확신도
                # exploration - 무작위 투자 유무
                action, confidence, exploration = \
                    self.agent.decide_action(
                        pred_value, pred_policy, epsilon)

                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                # 실제 행동 - 결정한 행동을 수행 하도록 에이전트의 act() 함수를 호출
                # act() 함수는 즉시 보상과 지연 보상을 반환
                immediate_reward, delayed_reward = \
                    self.agent.act(action, confidence)


                # 행동 및 행동에 대한 결과를 기억
                # (학습 데이터 샘플, 에이전트 행동, 즉시보상, 가치 신경망 출력, 정책 신경망 출력
                # 포트폴리오 가치, 보유 주식수, 탐험 위치)
                # 목적: 1. 배치 학습데이터, 2. 가시화기 차트 그릴때
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

                # 지연 보상 발생된 경우 미니 배치 학습
                # 지연 보상 = 임계치가 넘는 손익률이 발생 했을때.
                if learning and (delayed_reward != 0):
                    self.fit(delayed_reward, discount_factor)

            # 에포크 종료 후 학습(sample이 없는 경우)
            # 남은 미니 배치를 학습.
            if learning:
                self.fit(
                    self.agent.profitloss, discount_factor, full=True)

            # 에포크 관련 정보 로그 기록
            """
                num_epoches_digit - 주식 종목 코드
                epoch_str - 현재 에포크 번호
                time_end_epoch - 현재 에포크 시간(끝난 시간)
                elapsed_time_epoch - 에포크 수행 소요 시간
                loss - 학습 손실
                epsilon - 탐험률
                agent.num_buy - 매수 행동 수
                agent.num_sell - 매도 행동 수
                agent.num_stocks - 보유 주식 수
                learning_cnt - 미니 배치 수행 횟수
                
                
            """
            """
                *** 파이썬 팁 ***
                rjust() - 문자열 자릿수에 맞게 오른쪽으로 정렬해주는 함수.
                ex 1, "1".rjust(5)를 하면 '    1' - 앞에 빈칸 4자리를 채워주고
                1을 붙여서 5자리 문자열을 만들어 준다.
                ex 2, "1".rjust(5, '0') - '00001'
                as, ljust()
                
                format() 함수에서 키워드 명 뒤에 콜론(:)을 붙이고 형식 옵션을 지정할 수 있다.
                {:,.0f} - 천단위에 ,을 붙이고, 소수점 0째까지 표기.
                
            """
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            if self.learning_cnt > 0:
                self.loss /= self.learning_cnt
            logging.info("[{}][Epoch {}/{}] Epsilon:{:.4f} "
                "#Expl.:{}/{} #Buy:{} #Sell:{} #Hold:{} "
                "#Stocks:{} PV:{:,.0f} "
                "LC:{} Loss:{:.6f} ET:{:.4f}".format(
                    self.stock_code, epoch_str, num_epoches, epsilon, 
                    self.exploration_cnt, self.itr_cnt,
                    self.agent.num_buy, self.agent.num_sell, 
                    self.agent.num_hold, self.agent.num_stocks, 
                    self.agent.portfolio_value, self.learning_cnt, 
                    self.loss, elapsed_time_epoch))

            # 에포크 관련 정보 가시화
            self.visualize(epoch_str, num_epoches, epsilon)

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간(프로그램 전체 시간)
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        # 프로그램 전체 로그.
        with self.lock:
            logging.info("[{code}] Elapsed Time:{elapsed_time:.4f} "
                "Max PV:{max_pv:,.0f} #Win:{cnt_win}".format(
                code=self.stock_code, elapsed_time=elapsed_time, 
                max_pv=max_portfolio_value, cnt_win=epoch_win_cnt))
    # 정책 신경망, 가치 신경망을 확인하여 있는 경우 save.
    def save_models(self):
        if self.value_network is not None and \
                self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and \
                self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)


class DQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        # value_network_path 을 속성으로 저장 - models/{}.h5
        self.value_network_path = value_network_path
        # 가치 신경망 생성.
        self.init_value_network()

    """
        *** 파이썬 팁 ***
        리스트를 역으로 뒤집는 3가지 방법.
        lst.reverse()
        reverse(lst)
        lst[::01]
    """
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        # 메모리 배열을 역으로 묶어준다.
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        #샘플 배열 x, label 배열 y_value를 준비하고 배열은 모두 0으로 초기화
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        # 메모리를 역으로 취했기 때문에 for문은 배치 학습 데이터의 마지막 부분 부터 처리!.
        # JUN2 변경 가능으로 보임.
        for i, (sample, action, value, reward) in enumerate(memory):
            x[i] = sample
            y_value[i] = value
            # r = 마지막 손익률 - 현재 손익률 + 다음 행동 수행 시점 손익률 - 현재 손익률
            r = (delayed_reward + reward_next - reward * 2) * 100
            # 다음 상태의 최대 가치에 할인률을 적용해 구한 r을 더해준다.
            y_value[i, action] = r + discount_factor * value_max_next
            value_max_next = value.max()
            reward_next = reward
            #DQN Linear 부분에서는 정책 신경망을 다루지 않는다.
        return x, y_value, None


class PolicyGradientLearner(ReinforcementLearner):
    def __init__(self, *args, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        # value_network_path 을 속성으로 저장 - models/{}.h5
        self.policy_network_path = policy_network_path
        # 정책 신경망 load
        self.init_policy_network()

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        reward_next = self.memory_reward[-1]
        for i, (sample, action, policy, reward) in enumerate(memory):
            x[i] = sample
            y_policy[i] = policy
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_policy[i, action] = sigmoid(r)
            reward_next = reward
        return x, None, y_policy


class ActorCriticLearner(ReinforcementLearner):
    def __init__(self, *args, shared_network=None, 
        value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if shared_network is None:
            self.shared_network = Network.get_shared_network(
                net=self.net, num_steps=self.num_steps, 
                input_dim=self.num_features)
        else:
            self.shared_network = shared_network
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=shared_network)

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) \
            in enumerate(memory):
            x[i] = sample
            y_value[i] = value
            y_policy[i] = policy
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            y_policy[i, action] = sigmoid(value[action])
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, y_policy


class A2CLearner(ActorCriticLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) \
            in enumerate(memory):
            x[i] = sample
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            advantage = value[action] - value.mean()
            y_policy[i, action] = sigmoid(advantage)
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, y_policy


class A3CLearner(ReinforcementLearner):
    def __init__(self, *args, list_stock_code=None, 
        list_chart_data=None, list_training_data=None,
        list_min_trading_unit=None, list_max_trading_unit=None, 
        value_network_path=None, policy_network_path=None,
        **kwargs):
        assert len(list_training_data) > 0
        super().__init__(*args, **kwargs)
        self.num_features += list_training_data[0].shape[1]

        # 공유 신경망 생성
        self.shared_network = Network.get_shared_network(
            net=self.net, num_steps=self.num_steps, 
            input_dim=self.num_features)
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)

        # A2CLearner 생성
        self.learners = []
        for (stock_code, chart_data, training_data, 
            min_trading_unit, max_trading_unit) in zip(
                list_stock_code, list_chart_data, list_training_data,
                list_min_trading_unit, list_max_trading_unit
            ):
            learner = A2CLearner(*args, 
                stock_code=stock_code, chart_data=chart_data, 
                training_data=training_data,
                min_trading_unit=min_trading_unit, 
                max_trading_unit=max_trading_unit, 
                shared_network=self.shared_network,
                value_network=self.value_network,
                policy_network=self.policy_network, **kwargs)
            self.learners.append(learner)

    def run(
        self, num_epoches=100, balance=10000000,
        discount_factor=0.9, start_epsilon=0.5, learning=True):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(
                target=learner.run, daemon=True, kwargs={
                'num_epoches': num_epoches, 'balance': balance,
                'discount_factor': discount_factor, 
                'start_epsilon': start_epsilon,
                'learning': learning
            }))
        for thread in threads:
            thread.start()
            time.sleep(1)
        for thread in threads: thread.join()
