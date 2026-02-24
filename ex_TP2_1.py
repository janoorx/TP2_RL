import random

#Question 1
def step(state, action):
    """
    Docstring pour step
    
    :param state: Description
    :param action: Description

    :return: next_state, reward, terminal

    terminal est un booléen, True si c'est terminé, False sinon
    """
    if action=="hit":
        state["player"] += draw_card()
        if state["player"] < 1:
            return state, -1, True
        elif state["player"] > 21:
            return state, -1, True
        else:
            return state, 0, False
    elif action=="stick":
        #Le dealer joue tant qu'il a moins de 17 points
        while state["dealer"] < 17: 
            state["dealer"] += draw_card()

            #Dealer passe en dessous de 1, il perd
            if state["dealer"] < 1: 
                return state, 1, True
            
        #Dealer perd
        if state["dealer"] > 21 or state["player"] > state["dealer"]: 
            return state, 1, True
        #Dealer gagne
        elif state["player"] < state["dealer"]: 
            return state, -1, True
        #Egalité
        else: 
            return state, 0, True

# Question 2
def draw_card():
    """
    Docstring pour draw_card

    :return: card_value
    """
    value = random.randint(1, 10)
    sign = random.choice([-1, 1, 1]) # 1/3 chance de tirer une carte rouge, 2/3 chance de tirer une carte noire

    return sign * value

def init_game():
    """
    Docstring pour init_game

    :return: state
    """
    state = {}
    state["player"] = random.randint(1, 10)
    state["dealer"] = random.randint(1, 10)
    return state

#Game loop test
def ex_TP2_1():
    print
    state = init_game()
    print("Initial state:", state)

    terminal = False
    while not terminal:
        action = input("Enter action (hit/stick): ")
        if action not in ["hit", "stick"]:
            print("Invalid action. Please enter 'hit' or 'stick'.")
            continue

        next_state, reward, terminal = step(state, action)
        print("Next state:", next_state, "Reward:", reward, "Terminal:", terminal)

        state = next_state
