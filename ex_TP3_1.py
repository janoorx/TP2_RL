from ex_TP2_1 import *
from ex_TP2_2 import *
from ex_TP2_3 import *
from ex_TP3_2 import *

INTERVALLES_DEALER = [(1,4), (4,7), (7,10)]
INTERVALLES_PLAYER = [(1,6), (4,9), (7,12), (10,15), (13,18), (16,21)]
TAILLE_VECTEUR = 36
ALPHA = 0.01
EPSILON = 0.05
    
#Question 1
def feature_vector(state, action):
    """
    Construit le vecteur de caracteristiques binaires phi(s, a).
    Retourne un numpy array de dimension 36.
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

#! A revoir
def greedy_policy_const(state,theta):
    # dealer_i = state["dealer"] - 1
    # player_i = state["player"] - 1

    # if random.random() <= epsilon:
    #     return random.choice(ACTIONS)

    # # Calcul Qsa = ϕ(s, a)*θ
    

    # hit_value = Qsa[dealer_i, player_i, ACTION_TO_INDEX["hit"]]
    # stick_value = Qsa[dealer_i, player_i, ACTION_TO_INDEX["stick"]]
    # if hit_value > stick_value:
    #     return "hit"
    # else :
    #     return "stick"

    #Greedy policy : 
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
        
        # Calcul de la MSE à chaque épisode$
        MSE = calcul_MSE(Qsa,Qsa_MC)
        MSEs.append(MSE)

    return Qsa, MSEs


def test_SARSA_linear_variable_lambdas(N,lambda_values, Qsa_MC):
    MSEs = []
    for lambda_val in lambda_values:
        Qsa, _ = train_SARSA_linear(N,lambda_val,Qsa_MC)
        MSE = calcul_MSE(Qsa,Qsa_MC)
        MSEs.append(MSE)
        print("MSE pour lambda =", lambda_val, ":", MSE)
    
    return MSEs


def ex_TP3_1():
    N = 1000000
    N0 = 100
    N_lambda = 10
    #Calcul des différentes valeurs de lambda à tester
    lambda_values = [i / N_lambda for i in range(0, N_lambda + 1)] 

    Qsa_MC, _, _ = train_monte_carlo_control(N, N0) #On récupère uniquement Qsa du Monte Carlo pour le comparer avec SARSA(λ)
    MSEs_SARSA_lambda = test_SARSA_linear_variable_lambdas(N, lambda_values, Qsa_MC)
    plot_MSE_lambda(MSEs_SARSA_lambda, lambda_values)

if __name__ == "__main__":
    ex_TP3_1()