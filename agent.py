import numpy as np
import utils


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
    # TRADING_CHARGE = 0.00011  # 거래 수수료 0.011%
    # TRADING_CHARGE = 0  # 거래 수수료 미적용
    TRADING_TAX = 0.0025  # 거래세 0.25%
    # TRADING_TAX = 0  # 거래세 미적용

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(
        self, environment, min_trading_unit=1, max_trading_unit=2, 
        delayed_reward_threshold=.05):
        # Environment 객체
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        # 지연보상 임계치
        self.delayed_reward_threshold = delayed_reward_threshold

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0 
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상
        self.profitloss = 0  # 현재 손익
        self.base_profitloss = 0  # 직전 지연 보상 이후 손익
        self.exploration_base = 0  # 탐험 행동 결정 기준

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset_exploration(self):
        self.exploration_base = 0.5 + np.random.rand() / 2

    def set_balance(self, balance):
        self.initial_balance = balance
        """
        ratio_hold - 주식 보유비율
        (현재 들고 있는 주식수 / ( PV / 현재 주식 가격))
        PV / 현재 주식 가격 - 내가 현재 살수 있는 주식의 양
        
        ratio_portfolio_value - 포트 폴리오 가치 비율
        현재 PV / 학습 직전의 PV
        학습 직전 보다 내가 얼마나 벌거나 잃었나?
        """
    def get_states(self):
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = (
            self.portfolio_value / self.base_portfolio_value
        )
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    """
        입력으로 들어온 Epsilon의 확률로 무작위로 행동을 결정하고
        그렇지 않은 경우 신경망을 통해 행동을 결정합니다.
        
        0~1 사이의 랜덤 값을 생성하고 이값이 엡실론 보다 작으면 무작위로 행동을 결정
        만약 무작위로 행동을 결정 했다면, exploration_base를 새로 결정하고,
        exploration_base가 1에 가까우면 탐험할 때 매수를 선택 
        0에 가까우면 매도를 결정
    """
    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.

        """
            정책 확인 - 매수, 매도 결정
            없는 경우 매수, 매도 정책 률 확인
            없는 경우 탐험
            같은 경우 탐험
            있는 경우 그값으로 예측
        """
        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            # 매수, 매도의 예측이 같은 경우.
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

        # 탐험 결정
        """
            1. 탐험 결정
             - exploration_base가 random 값보다 크다.
              - buy
             - 작다
              - sell
            2. 신경망 결정
            np.argmax(axis) - axis 중 가장 큰 값의 인덱스들을 반환하는 함수.
        """
        if np.random.rand() < epsilon:
            exploration = True
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY
            else:
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    """
        신용, 공매도 (X)
        
        불가능 요소
        1. 매수 결정 잔금 부족
        2. 매도 결정 보유 주식 (X)
    """
    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (
                1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인 
            if self.num_stocks <= 0:
                return False
        return True

    """
        정책 신경망이 결정한 행동의 신뢰가 높을 수록 매수 or 매도하는 단위를 크게.
        
        높은 신뢰로 매수 결정 더 많은 주식 매수, 높은 신뢰 매도 더 많은 주식 매도.
    """
    def decide_trading_unit(self, confidence):
        # confidence 없는 경우 최소 거래
        if np.isnan(confidence):
            return self.min_trading_unit
        added_traiding = max(min(
            int(confidence * (self.max_trading_unit - 
                self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_traiding


    """
        에이전트가 결정한 행동을 수행한다.
        
        action - 탐험 or 정책 신경망을 통해 결정한 행동으로 매수, 매도를 의미하는 0 or 1의 값
        
        cofidence - 정책 신경망을 통해 결정한 경우 결정한 행동에 대한 소프트맥스 확률값
        
        먼저 이 행동을 할 수 있는지 확인하고 할 수 없는 경우 아무 행동도 하지 않게 광망(hold)합니다.
        그리고, 환경 객체에서 현재 주가를 받아 옵니다.     
        이가격은 매수 금액, 매도 금액, 포트폴리오 가치를 계산할 때 사용합니다. 
        즉시 보상은 에이전트가 행동 할 때 마다 결정 되기 때문에 초기화합니다. 
    
    """
    def act(self, action, confidence):
        # action이 가능 하지 않으면 HOLD
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화 - 즉시 보상은 매번 일어 나기 때문에 초기화 필요.
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단 - confidence에 따라 단위 결정
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                self.balance - curr_price * (1 + self.TRADING_CHARGE) \
                    * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (
                            curr_price * (1 + self.TRADING_CHARGE))),
                        self.max_trading_unit
                    ),
                    self.min_trading_unit
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) \
                * trading_unit
            if invest_amount > 0:
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) \
                    * trading_unit
            if invest_amount > 0:
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        # 포트폴리오 가치 갱신
        # pv = 현금 + 주식 가격 * 주식 수
        self.portfolio_value = self.balance + curr_price \
            * self.num_stocks
        # 현재까지 PV 변동 비율
        self.profitloss = (
            (self.portfolio_value - self.initial_balance) \
                / self.initial_balance
        )
        
        # 즉시 보상 - 수익률
        self.immediate_reward = self.profitloss

        # 지연 보상 - 익절, 손절 기준
        #
        delayed_reward = 0
        self.base_profitloss = (
            (self.portfolio_value - self.base_portfolio_value) \
                / self.base_portfolio_value
        )
        # 지연 보상이 정해 놓은 익절, 손절 기준보다 크거나 작아지면
        # 지연 보상값에 즉시 보상을 넣는다.
        # 여기부분 이상 JUN1
        if self.base_profitloss > self.delayed_reward_threshold or \
            self.base_profitloss < -self.delayed_reward_threshold:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0

        return self.immediate_reward, delayed_reward
