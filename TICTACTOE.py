#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:27:26 2023

@author: myco
"""

import random, timeit
from collections import namedtuple
from typing import Iterator, Hashable, List, Dict
__author__ = 'Henry Gotjen'
__email__ = 'hgotjen@gmail.com'
__version__ = '0.1.0'


def shell_type() -> str:
    try:
        return get_ipython().__class__.__name__
        # Jupyter notebook or qtconsole:    'ZMQInteractiveShell'
        # Terminal running IPython:         'TerminalInteractiveShell'
        # Spyder running IPython:           'SpyderShell
    except NameError:
        return 'Python Interpreter'

ISIPYTHON = shell_type() in ['TerminalInteractiveShell', 'SpyderShell']

''' Player convention represented by bools '''
Player = bool
PLAYER = [X, O] = [True, False]
PlayerName = {True: 'X', False: 'O', None: ' '}

''' Square index representing squares 0-8
 0 | 1 | 2 
-----------
 3 | 4 | 5 
-----------
 6 | 7 | 8 
'''
Square = int
SQUARE = list(range(9))

''' BitBoard representation of TTT boards with positional masks '''
''' For examples 0b000111000 masks squares [3,4,5]'''
BitBoard = int
BB_NONE = 0
BB_ALL = 0x1ff
BB_SQUARE = [1 << sq for sq in SQUARE]

# Winning track masks
BB_ROW = [0x7 << 3*ii for ii in range(3)]
BB_COL = [0x49 << ii for ii in range(3)]
BB_DIAG = [0x54, 0x111]
BB_WINS = BB_ROW + BB_COL + BB_DIAG

''' TTT Outcome types '''
Outcome = int
OUTCOMES = [ActiveGame, Draw, WinX, WinO] = [None, 0, 1, -1]
EvalLookup = {X: 1, O: -1, None: 0}
Outcome = namedtuple('Outcome', 'Outcome Player BBWIN')
OutcomeName = {ActiveGame: 'Active Game in Progress',
               Draw: 'Drawn Game :\'(', WinX: 'X is the WINNER!', WinO: 'O is the WINNER!'}
DrawnOutcome = Outcome(Draw, None, 0)
ActiveOutcome = Outcome(ActiveGame, None, 0)

''' TTT Move type and input styles '''
Move = namedtuple('Move', 'Player Square')
nullmove = Move(None,None)
qezcMoves = 'qweasdzxc'
numpadMoves = '789456123'

''' Bitwise functions '''


def msb(bb: BitBoard) -> Square:
    '''Returns square id from the left side of BitBoard'''
    return bb.bit_length()-1


def lsb(bb: BitBoard) -> Square:
    '''Returns square id from the right side of BitBoard'''
    return (bb & -bb).bit_length() - 1


def scanbb(bb: BitBoard) -> Iterator[Square]:
    '''Scans the BB mask from LSB to MSB yielding square id'''
    while bb:
        rem = msb(bb)
        yield rem
        bb ^= BB_SQUARE[rem]


def checkOverlap(BB1, BB2, mode:str='all') -> bool:
    ''' Checks for complete overlap between occupied squares and the win mask '''
    if mode=='all':
        for sq in scanbb(BB2):
            if not BB_SQUARE[sq] & BB1:
                return False
        return True
    elif mode=='any':
        return not BB1 & BB2

''' Move inputs and validation '''
def validMove(move: Move, throw: bool = False) -> bool:
    if throw:
        if not isinstance(move, Move):
            raise InvalidMoveError(f'move is not Move! {move}')
        if not isinstance(move.Player, bool):
            raise InvalidMoveError('Player must be boolean')
        if not move.Square in range(9):
            raise InvalidMoveError('Square must be 0-8')
    return isinstance(move, Move) and isinstance(move.Player, bool) and move.Square in range(9)


def qezc2sq(char) -> Square:
    if char not in qezcMoves:
        raise InvalidMoveError('qezc entry not valid')
    return qezcMoves.index(char)


def numpad2sq(char) -> Square:
    if char not in numpadMoves:
        raise InvalidMoveError('numpad entry not valid')
    return numpadMoves.index(char)


''' Exception Types '''


class InvalidMoveError(ValueError):
    '''Raised when a move has invalid syntax'''


class IllegalMoveError(ValueError):
    '''Raised when an attempted move is illegal in the position'''


''' Main Types '''
class T3position:
    def __init__(self,board):
        self.BB_X = board.BB_X
        self.BB_O = board.BB_O
        self.turn = board.turn
    def restore(self,board):
        board.BB_X = self.BB_X
        board.BB_O = self.BB_O
        board.turn = self.turn
        
class T3board:
    '''Board to track the state of TicTacToeBoard'''
    BB_X: int
    BB_O: int
    turn: bool
    move_stack: List[Move]
    outcome: Outcome = None
    move_validation:dict = {'player': True, 'occupation': True}

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # set board occupancy to none
        self.BB_X = BB_NONE
        self.BB_O = BB_NONE

        # set turn to X
        self.turn = X 
        self.move_stack = []
        self.outcome = ActiveOutcome

    @property
    def BB_OCC(self) -> BitBoard:
        return self.BB_X ^ self.BB_O
    @property
    def state(self) -> T3position:
        return T3position(self)

    def validatemove(self, move):
        # Check that this move has valid values
        validMove(move, throw=True)

        # Check that the correct player is making a move
        if self.move_validation['player'] and self.turn ^ move.Player:
            raise IllegalMoveError(f'It is not {PlayerName[move.Player]}\'s turn')

        # Check the target square is unoccupied
        if self.move_validation['occupation'] and (plyr:=self.atSquare(move.Square)) is not None:
            raise IllegalMoveError(f'Square {move.Square} is occupied by {PlayerName[plyr]}')
        
    def push(self, move: Move) -> None:
        ''' Pushes a move optject to the board by:
                1. Check move validity and legality
                2.  [optional] validating player turn
                3.  Checking move square is unoccupied
                4. Adding move to self.move_stack
                5. Placing move on board state
                6. Switching turn
                7. Updating game outcome (Win, Draw, Activegame)
        '''
        
        self.validatemove(move)
        
        # Make the move and add to stack
        if move.Player:
            self.BB_X ^= BB_SQUARE[move.Square]
        else:
            self.BB_O ^= BB_SQUARE[move.Square]
            
        self.move_stack.append(move)
        self.turn = not move.Player
        
        if not self.outcome.Outcome:
            # skip updateOutcome if one already exists
            # allows player to keep making moves without changing the outcome.
            self.updateOutcome()

    def push_movesq(self, sq) -> None:
        self.push(Move(self.turn, sq))

    def pop(self) -> Move:
        ''' Removes a move from the board stack and restores the state of the board prior to the move '''
        assert len(self.move_stack) > 0, 'No moves to pop!'

        move = self.move_stack.pop()

        if move.Player:
            self.BB_X ^= BB_SQUARE[move.Square]
        else:
            self.BB_O ^= BB_SQUARE[move.Square]

        self.turn = not self.turn
        self.updateOutcome()
        return move
    
    def playerhaswin(self):
        ''' check if there are remain win paths on the board for player '''
            
        Xhaswin = False
        Ohaswin = False
        for BB_WIN in BB_WINS:
            if not Xhaswin: Xhaswin = checkOverlap(self.BB_O, BB_WIN,'any')
            if not Ohaswin: Ohaswin = checkOverlap(self.BB_X, BB_WIN,'any')
            if Xhaswin and Ohaswin: break
        return (Xhaswin,Ohaswin)

    def updateOutcome(self) -> None:
        ''' Test for game terminating outcome '''

        # Check overlap of winning tracks with X, and O occupancy masks
        for bbwin in BB_WINS:
            if checkOverlap(self.BB_X, bbwin):
                # X has won
                self.outcome = Outcome(WinX, X, bbwin)
                return
            if checkOverlap(self.BB_O, bbwin):
                # 0 has wone
                self.outcome = Outcome(WinO, O, bbwin)
                return

        # If board is full and no tracks were winning, its a draw
        if not self.BB_OCC ^ BB_ALL:  # len(self.move_stack)==9:
            # Drawn
            self.outcome = DrawnOutcome
            return

        self.outcome = ActiveOutcome

    def terminated(self):
        return self.outcome.Outcome is not ActiveGame

    def fanfare(self):
        print(OutcomeName[self.outcome.Outcome])

    def _transpose_key(self) -> Hashable:
        return (self.BB_X, self.BB_O, self.turn)
    
    def isvalid(self):
        assert not self.BB_X & self.BB_O, 'Overlaping X and O occupation'
        return True

    def ListLegalMoves(self) -> List[Move]:
        return list(self.GenerateLegalMoves())
    
    def GenerateLegalMoves(self) -> Iterator[Move]:
        for sq in self.OpenSquares():
            yield Move(self.turn, sq)

    def OpenSquares(self) -> Iterator[Square]:
        yield from scanbb(self.BB_OCC ^ BB_ALL)

    def square_stack(self) -> list[Square]:
        return [move.Square for move in self.move_stack]
    
    def isfull(self) -> bool:
        ''' returns True when all squares are occupied '''
        return not (self.BB_OCC ^ BB_ALL)
    
    def atSquare(self, sq) -> bool:
        ''' returns the player (X,O) at 'sq', otherwise None'''
        if self.BB_X & BB_SQUARE[sq]:
            return True
        elif self.BB_O & BB_SQUARE[sq]:
            return False
        else:
            return None

    def show(self) -> None:
        print(self)

    def rawStr(self) -> Iterator[str]:
        for sq in SQUARE:
            s = PlayerName[self.atSquare(sq)]
            if self.move_stack and ISIPYTHON and self.move_stack[-1].Square == sq:
                s = '\x1b[1;31m'+s+'\x1b[0m'
            yield s
    
    def __str__(self) -> str:
        brk = '\n ----------- \n'
        line = '  {} | {} | {}  '
        fmt = line + brk + line + brk + line
        return fmt.format(*self.rawStr())
    
    @classmethod
    def FromMoveOrder(cls, stack):
        board = cls()
        for move in stack:
            if isinstance(move, Move):
                board.push(move)
            else:
                board.push_movesq(move)
        return board

SuperMove = namedtuple('SuperMove', 'Player Target Square')

def validSuperMove(move: Move, throw: bool = False) -> bool:
    if throw:
        assert isinstance(move, SuperMove), f'move is not SuperMove! {move}'
        if not isinstance(move.Player, bool):
            raise InvalidMoveError('Player must be boolean')
        if not move.Target in range(9):
            raise InvalidMoveError('Target must be 0-8')
        if not move.Square in range(9):
            raise InvalidMoveError('Square must be 0-8')
    return isinstance(move, Move) and isinstance(move.Player, bool) and move.Square in range(9)

class superT3board(T3board):
    sub:list[T3board]
    target:int = None
    move_validation:dict = {'player': True, 'target': True}
    
    def __init__(self) -> None:
        # init sub boards - each is a tttboard
        self.sub = [T3board() for _ in range(9)]
        self.reset()
    
    def validateMove(self,move):
        # Move Validation
        validSuperMove(move,throw=True)
        
        if self.move_validation['player'] and self.turn ^ move.Player:
            raise IllegalMoveError('It\'s not your turn man.')
        
        validTarget = (self.target is None or self.target == move.Target)
        if self.move_validation['target'] and not validTarget:
            raise IllegalMoveError(f'Move must target cell {self.target}. Not {move.Target}.')
            
        # Ensure move square is unoccupied
        if self.atSquare(move.Square,move.Target) is not None:
            raise IllegalMoveError(f'Square {move.Square} in target {move.Target} is occupied.')
        
    def push(self,move:SuperMove) -> None:
        # move validation
        self.validateMove(move)
        
        # Push move and add to stack
        submv = Move(self.turn,move.Square)
        self.sub[move.Target].push(submv)
        self.move_stack.append(move)
        
        # Update
        self.target = move.Square
        if self.sub[move.Target].isfull() \
            or self.sub[move.Target].terminated():
            ''' no squares available in target, open choice'''
            self.target = None
        
        self.turn = not self.turn
        self.updateSuperOutcome()
        
    def push_movesq(self,square:int,target:int=None) -> None:
        if target is None:
            if self.target is None:
                raise InvalidMoveError('Must specify target cell in this case')
            target = self.target
        move = SuperMove(self.turn,target,square)
        self.push(move)
        
    def pop(self) -> SuperMove:
        assert len(self.move_stack) > 0, 'No moves to pop!'

        move = self.move_stack.pop()
        self.sub[move.Target].pop()
        
        if self.move_stack:
            self.target = self.move_stack[-1].Square
        else:
            self.target = None
            
        self.updateSuperOutcome()
        return move  
    
    def atSquare(self,square,target):
        return self.sub[target].atSquare(square)
    
    def updateSuperOutcome(self) -> None:
        ''' Update Superboard Position from subboard outcomes'''
        self.BB_X = 0
        self.BB_O = 0
        for ii,sub in enumerate(self.sub):
            sub.updateOutcome()
            if sub.outcome.Player is True:
                self.BB_X ^= BB_SQUARE[ii]
            elif sub.outcome.Player is False:
                self.BB_O ^= BB_SQUARE[ii]
        
        self.updateOutcome()
    
    def GenerateLegalMoves(self) -> Iterator[SuperMove]:
        if self.target is None:
            targets = self.OpenTargets()
        else:
            targets = [self.target]
        
        for target in targets:
            for square in self.sub[target].OpenSquares():
                yield SuperMove(self.turn,target,square)
                
    def OpenTargets(self) -> Iterator[Square]:
        for sq in self.OpenSquares():
            if not self.sub[sq].terminated():
                yield sq
                
    def reset(self) -> None:
        for board in self.sub:
            board.reset()
        self.target = None
        
        # set board occupancy to none
        self.BB_X = BB_NONE
        self.BB_O = BB_NONE

        # set turn to X
        self.turn = X 
        self.move_stack = []
        self.outcome = ActiveOutcome
        
    def get_sub_strings(self):
        sx = [r'             ',
              r'  \\    //   ',
              r'    \\//     ',
              r'    //\\     ',
              r'  //    \\   ']
        so = [r'    ______   ',
              r'   //    \\  ',
              r'  ||     ||  ',
              r'  \\____//   ',
              r'             ']
        sd = [r'             ',
              r'             ',
              r'    drawn    ',
              r'             ',
              r'             ']
        builder = []
        for sub in self.sub:
            if sub.terminated():
                if sub.outcome.Player is True:
                    builder.append(sx)
                elif sub.outcome.Player is False:
                    builder.append(so)
                else:
                    builder.append(sd)
            else:
                builder.append(sub.__str__().split('\n'))
        return builder
    
    def __str__(self) -> str:
        substr = self.get_sub_strings()
        
        if self.target is not None and ISIPYTHON:
            substr[self.target] = ['\x1b[1;32m'+s+'\x1b[0m' for s in substr[self.target]]
        
        divv = ' || '
        divh = '-' * 47
        divh = '\n' + divh + '\n' + divh + '\n'
        
        builder = 'SUPER TICTACTOE BOARD\n'
        for k in range(3):
            for j in range(5):
                for i in range(3):
                    builder += substr[i+3*k][j]
                    if i<2: builder += divv
                if j<4: builder += '\n'
            if k<2: builder += divh
            
        if self.target and not ISIPYTHON:
            builder += f'\nTarget cell {self.target}\n' + self.sub[self.target].__str__()
            
        return builder

class T3engine:
    '''Peaks ahead in a ttt game'''
    board: T3board = None
    mem: Dict = {}
    
    def __init__(self, board: T3board = T3board()) -> None:
        self.board = board
        self.reusecnt = 0

    def Search(self, debug: bool = False):
        if self.board.terminated():
            engineMove = None
            return EvalLookup[self.board.outcome.Player], []
        # elif self.board.BB_OCC == 0:
        #     return 0,Move(self.board.turn, 0)
        
        Evaluation, engineMove = self._search(debug=debug)
        return (Evaluation, engineMove)

    def _search(self, depth: int = 0, debug: bool = False):
        state = self.board._transpose_key()
        if state in self.mem:
            self.reusecnt += 1
            return self.mem[state]
        elif self.board.terminated():
            Evaluation = 1000*self.board.outcome.Outcome / depth
            if debug and False:
                self.board.show()
                print(' '*depth, self.board.move_stack[-1], f'Eval: {Evaluation}')
            return Evaluation,None

        BEST = None
        for move in self.board.GenerateLegalMoves():
            self.board.push(move)
            lasteval = self._search(depth=depth+1, debug=debug) 
            self.board.pop()
            if BEST is None or\
                (self.board.turn and lasteval[0] > BEST[0]) or \
                (not self.board.turn and lasteval[0] < BEST[0]):
                BEST = (lasteval[0], move)
        
        # ieval = range(len(evals))
        # if self.board.turn:
        #     ibest = max(ieval,key=lambda ii: evals[ii])
        # else:
        #     ibest = min(ieval,key=lambda ii: evals[ii])
        
        # best = (evals[ibest][0], moves[ibest])
        # self.mem[state] = best
        # if not best == best:
        #     pass
        # return best
        return BEST

    def depthFirstCount(self, depth: int = 9) -> int:
        if depth == 0 or self.board.terminated():
            return 1

        poscount = 0
        for move in board.GenerateLegalMoves():
            # print('dink')
            self.board.push(move)
            poscount += self.depthFirstCount(depth-1)
            self.board.pop()

        return poscount

def keyplayT3(board: T3board = T3board()) -> T3board:
    moveLookup = qezc2sq
    while not board.terminated():
        try:
            print(board)
            c = input('Enter Move for {}: '.format(PlayerName[board.turn]))
            sq = moveLookup(c)

            board.push(Move(board.turn, sq))

        except Exception as err:
            print(board)
            print(err)
            return board

    print(board)
    print(OutcomeName[board.outcome.Outcome])

    return board

def keyplayT3engine(board: T3board = T3board(), side=X) -> T3board:
    eng = T3engine(board)
    moveLookup = qezc2sq
    while not board.terminated():
        Evaluation, engineMove = eng.Search(debug = False)
        print(f'Evaluation: {Evaluation}')
        print(f'Best Move for {PlayerName[board.turn]}: {engineMove.Square}')
        if board.turn == side:
            ''' Human player turn '''
            print(board)
            c = input('Enter Move for {}: '.format(PlayerName[board.turn]))

            try:
                sq = moveLookup(c)
            except Exception:
                break

            board.push(Move(board.turn, sq))
        else:
            ''' Engine player side '''
            board.push(engineMove)

    print(board)
    print(OutcomeName[board.outcome.Outcome])

    return eng, board

def keyplaysuperT3(board:superT3board = superT3board()) -> superT3board:
    
    while not board.terminated():
        board.show()
        try:
            if board.target is None:
                c = input(f'{PlayerName[board.turn]}, choose target cell: ')
                board.target = qezc2sq(c)
            c = input(f'{PlayerName[board.turn]}, enter move: ')
            board.push_movesq(qezc2sq(c))  
        except KeyboardInterrupt:
            break
        except Exception as err:
            print(err) 
            break
    
    return board   

T3AGENTTYPES = ['random','human','engine']
class T3agent:
    agenttype:str
    side: bool = None
    board = None
    engine = None
    issuper:bool = False
    def __init__(self,agenttype:str, side:bool, board, engine=None) -> None:
        assert agenttype in T3AGENTTYPES, f'Invalid Agent {agenttype}'
        # assert side in [X,O], 'Side must be X or O'
        self.agenttype = agenttype.lower()
        self.side = side
        self.set_board(board)
        self.set_engine(engine)
        
        self.intro
    
    def set_board(self,board) -> None:
        # Validate board
        if isinstance(board,superT3board):
            self.issuper = True
        elif isinstance(board,T3board):
            self.issuper = False
        else:
            ValueError('Board must be T3board or superT3board')
        
        self.board = board
        
    def set_engine(self,engine) -> None:
        # Validate board
        if self.agenttype == 'engine':
            if engine is None:
                self.engine = T3engine(self.board)
            elif isinstance(engine,T3engine):
                self.engine = engine
            else:
                raise ValueError('Engine must be T3engine or None')
        else:
            self.engine = None
        
    def intro(self,disp:bool=True):
        s = f'{self.agenttype} playing for {PlayerName[self.side]}'
        if disp:    print(s)
        else:       return s
    
    def get_move_random(self):
        movelist = list(self.board.GenerateLegalMoves())
        if not movelist: # no moves
            return None
        return random.choice(movelist)
    
    def get_move_human(self):
        self.board.show()
        try:
            if self.issuper:
                if self.board.target is None:
                    t = input(f'{PlayerName[self.board.turn]}, choose target: ')
                    target = qezc2sq(t)
                else:
                    target = self.board.target
            c = input(f'{PlayerName[self.board.turn]}, enter move: ')
            square = qezc2sq(c)
        except InvalidMoveError:
            return None
        except KeyboardInterrupt:
            return None
        
        if self.issuper:
            return SuperMove(self.board.turn, target, square)
        else:
            return Move(self.board.turn, square)
        
    def get_move_engine(self,*a,**k):
        evalulation, enginemove = self.engine.Search(*a,**k)
        return enginemove
        
    def get_move(self, *a,**k):
        if self.agenttype == 'random':
            return self.get_move_random()
        elif self.agenttype == 'human':
            return self.get_move_human()
        elif self.agenttype == 'engine':
            return self.get_move_engine(*a,**k)
     
def T3match(player1,player2, board=None,engine=None,pause:bool=False):
    if board is None:
        board = T3board()
    if engine is None and 'engine' in [player1, player2]:
        engine = T3engine(board)
    
    agents = {X: T3agent(player1, side=X, board=board, engine=engine),
              O: T3agent(player2, side=O, board=board, engine=engine)}
    try:
        while not board.terminated():
            move = agents[board.turn].get_move()
            print(f'{PlayerName[board.turn]} ({agents[board.turn].agenttype}) plays {move}')
            if move is None:
                print('Play Terminated by invalid move')
                return board, engine, agents
            try:
                board.push(move)
            except IllegalMoveError as err:
                print(err)
                print('Play terminated by illegal move')
            if pause:
                board.show()
                input()
    except KeyboardInterrupt:
        pass
    
    board.show()
    board.fanfare()   
    return board, engine, agents

def testT3match():
    return  T3match('human','engine',board = board,pause=True)

def testSuperT3poscnt():
    board = superT3board()
    eng = T3engine(board)
    
    movecnt =[]
    tmax = 20
    for k in range(1,9*9+1):
        tic = timeit.timeit()
        movecnt.append( eng.depthFirstCount(k) )
        toc = timeit.timeit()
        print(k,': ', movecnt[-1])
        
        if (toc-tic) > tmax:
            break

if __name__ == '__main__':
    
    board = superT3board()