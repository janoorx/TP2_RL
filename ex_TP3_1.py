from ex_TP2_1 import *
from ex_TP2_2 import *
from ex_TP2_3 import *
from ex_TP3_2 import *

INTERVALLES_DEALER = [(1,4), (4,7), (7,10)]
INTERVALLES_PLAYER = [(1,6), (4,9), (7,12), (10,15), (13,18), (16,21)]
TAILLE_VECTEUR = 36
ALPHA = 0.01
EPSILON = 0.05


def compute_Qsa_from_theta(theta):
    """
    Calcule la table de valeurs d'action Q(s, a) à partir du vecteur de paramètres theta.
    Args:
        theta (np.array): vecteur de paramètres appris par l'algorithme de contrôle SARSA(λ)
    Returns:
        Qsa (np.array): table de valeurs d'action Q(s, a) calculée à partir de theta
    """
    Qsa = np.zeros((N_DEALER, N_PLAYER, len(ACTIONS)))
    for dealer in range(1, N_DEALER + 1):
        for player in range(1, N_PLAYER + 1):
            state = {"dealer": dealer, "player": player}
            for action in ACTIONS:
                Qsa[dealer - 1, player - 1, ACTION_TO_INDEX[action]] = np.dot(feature_vector(state, action), theta)
    return Qsa
    
#Question 1
def feature_vector(state, action):
    """
    Construit le vecteur de caracteristiques binaires phi(s, a).
    
    Args:
        state (dict): état actuel du jeu, avec les clés "player" et "dealer"
        action (str): action choisie ("hit" ou "stick")
    Returns:
        np.array: vecteur de caracteristiques binaires phi(s, a)
    """
    phi = np.zeros((len(INTERVALLES_DEALER),len(INTERVALLES_PLAYER),len(ACTIONS)))
    
    for i, int_dealer in enumerate(INTERVALLES_DEALER) : 
        if(state["dealer"]>=int_dealer[0] and state["dealer"]<=int_dealer[1]):

            for j, int_player in enumerate(INTERVALLES_PLAYER) : 
                if(state["player"]>=int_player[0] and state["player"]<=int_player[1]):

                    for k, action_i in enumerate(ACTIONS):
                        if(action_i == action):

                            phi[i,j,k] = 1

    return phi.flatten()

def greedy_policy_const(state,theta):
    """
    Politique epsilon-greedy constante basée sur le vecteur de paramètres theta.
    Args:       state (dict): état actuel du jeu, avec les clés "player" et "dealer
        theta (np.array): vecteur de paramètres appris par l'algorithme de contrôle SARSA(λ)
    Returns:      str: action choisie ("hit" ou "stick")
    """
    if random.random() <= EPSILON:
        return random.choice(ACTIONS)
    Q_values = []
    for action in ACTIONS:
        phi_sa = feature_vector(state, action)
        Q_value = np.dot(phi_sa, theta)
        Q_values.append(Q_value)
    best_action_index = np.argmax(Q_values)
    return ACTIONS[best_action_index]


#Question 2
def train_SARSA_linear(N,lambda_value, Qsa_MC):
     #MSEs à chaqeu épisode d'apprentissage
    MSEs = []
    
    theta = np.zeros((TAILLE_VECTEUR))
    print("Début de l'apprentissage SARSA avec",N,"épisodes et lambda = ", lambda_value)
   
    for _ in range(N):
        Esa = np.zeros((TAILLE_VECTEUR)) #Initialisation des traces d’éligibilité à 0

        state = init_game()
        terminal = False

        # Choisir la première action
        action = greedy_policy_const( state, theta)

        while not terminal:
            # dealer_i = state["dealer"] - 1
            # player_i = state["player"] - 1
            # action_i = ACTION_TO_INDEX[action]
            phi = feature_vector(state, action)
            # Exécuter l'action
            state_next, reward, terminal = step(state, action)

            # Choisir l'action suivante
            Qsa = np.dot(phi, theta)
            if terminal:
                delta = reward - Qsa
                action_next = None
                next_Qsa = 0.0
            else:
                action_next = greedy_policy_const(state_next, theta)
                next_Qsa = np.dot(feature_vector(state_next, action_next), theta)
                delta = reward + next_Qsa - Qsa
            
            #Des traces d’éligibilité accumulées : et ← γλ et−1 + ϕ(st, at).
            # — La mise à jour des paramètres : θ ← θ + α · δt · et
            Esa = Esa * lambda_value + phi
            theta += ALPHA * delta * Esa #Mise à jour globale de theta

            state = state_next
            if not terminal:
                action = action_next
        
        # Calcul de la MSE à chaque épisode (sur tous les (s,a))
        Qsa_hat = compute_Qsa_from_theta(theta)
        MSE = calcul_MSE(Qsa_hat, Qsa_MC)
        MSEs.append(MSE)

    return theta, MSEs


def test_SARSA_linear_variable_lambdas(N,lambda_values, Qsa_MC):
    MSEs = []
    for lambda_val in lambda_values:
        theta, _ = train_SARSA_linear(N,lambda_val,Qsa_MC)
        Qsa_hat = compute_Qsa_from_theta(theta)
        MSE = calcul_MSE(Qsa_hat, Qsa_MC)
        MSEs.append(MSE)
        print("MSE pour lambda =", lambda_val, ":", MSE)
    
    return MSEs

def ex_TP3_1():
    # Nombre d'épisodes d'apprentissage
    N = 10000
    # Paramètre de contrôle de l'exploration
    N0 = 100
    # Nombre de valeurs de lambda à tester
    N_lambda = 10
    Qsa_MC, _, _ = train_monte_carlo_control(N, N0) #On récupère uniquement Qsa du Monte Carlo pour le comparer avec SARSA(λ)
    
    choice = input("\033[1;36mVoulez-vous exécuter la question 2 (Différents lambdas) ou la question 3 (MSE) ou quitter l'exercice 'q'? (2, 3 ou q) : \033[0;37m")
    if choice == "2":
        lambda_values = [i / N_lambda for i in range(0, N_lambda + 1)] 
        MSEs_SARSA_lambda = test_SARSA_linear_variable_lambdas(N, lambda_values, Qsa_MC)
        plot_MSE_lambda(MSEs_SARSA_lambda, lambda_values)
    elif choice == "3":  
        MSEs = []
            
        lambda_test = 0
        _, MSEs_0 = train_SARSA_linear(N,lambda_test,Qsa_MC)
        MSEs.append((lambda_test, MSEs_0))

        lambda_test = 1
        _, MSEs_1 = train_SARSA_linear(N,lambda_test,Qsa_MC)
        MSEs.append((lambda_test, MSEs_1))

        plot_MSE(MSEs)
    elif choice == "q":
        print("\033[1;32mVous avez quitté l'exercice 1 du TP3.\033[0;37m")
    else:
        print("\033[1;31mChoix invalide. Veuillez entrer 2 ou 3.\033[0;37m")

if __name__ == "__main__":
    ex_TP3_1()