import random

#Question 1
def step(state, action):
    """ 
    Effectue une action dans le jeu de Blackjack.

    Args:
        state: état actuel du jeu, un dictionnaire avec les clés "player" et "dealer" représentant les points actuels du joueur et du dealer
        action: action choisie par le joueur, soit "hit" pour tirer une carte, soit "stick" pour rester

    Returns:
        next_state: état suivant du jeu après l'action, un dictionnaire avec les clés "player" et "dealer" représentant les points actuels du joueur et du dealer
        reward: récompense obtenue après l'action, un entier qui peut être -1 (perte), 0 (égalité) ou 1 (gain)
        terminal: booléen indiquant si le jeu est terminé (True) ou non (False)
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
    Tire une carte aléatoire. Les cartes ont une valeur entre 1 et 10, et peuvent être rouges (valeur négative) ou noires (valeur positive).

    Returns: 
        int: la valeur de la carte tirée, qui peut être négative (carte rouge) ou positive (carte noire)
    """
    value = random.randint(1, 10)
    sign = random.choice([-1, 1, 1]) # 1/3 chance de tirer une carte rouge, 2/3 chance de tirer une carte noire

    return sign * value

def init_game():
    """
    Initialise le jeu en tirant une carte pour le joueur et une pour le dealer.

    Returns:
        state: un dictionnaire avec les clés "player" et "dealer" représentant les points
    """
    state = {}
    state["player"] = random.randint(1, 10)
    state["dealer"] = random.randint(1, 10)
    return state

#Game loop test
def ex_TP2_1():
    """
    Simule une partie de Blackjack en utilisant les fonctions définies précédemment. 
    Le joueur peut entrer des actions ("hit" ou "stick") et le jeu affiche l'état suivant, 
    la récompense obtenue et si le jeu est terminé ou non.
    """
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
