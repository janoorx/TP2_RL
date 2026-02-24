from ex_TP2_1 import *
from ex_TP2_2 import *
from ex_TP2_3 import *
from ex_TP3_2 import *

INTERVALLES_DEALER = [(1,4), (4,7), (7,10)]
INTERVALLES_PLAYER = [(1,6), (4,9), (7,12), (10,15), (13,18), (16,21)]
    
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
def greedy_policy_const(epsilon,state,phi,theta):
    dealer_i = state["dealer"] - 1
    player_i = state["player"] - 1
    epsilon = N0 / (N0 + Ns[dealer_i, player_i])

    if random.random() <= epsilon:
        return random.choice(ACTIONS)

    # Greedy action on Q(s, a)
    hit_value = Qsa[dealer_i, player_i, ACTION_TO_INDEX["hit"]]
    stick_value = Qsa[dealer_i, player_i, ACTION_TO_INDEX["stick"]]
    if hit_value > stick_value:
        return "hit"
    else :
        return "stick"

    


#Question 2
def train_SARSA_linear(N,N0,lambda_value, Qsa_MC):
     #MSEs à chaqeu épisode d'apprentissage
    MSEs = []
    epsilon = 0.05
    alpha = 0.01
    theta = np.zeros((len(INTERVALLES_DEALER)*len(INTERVALLES_PLAYER)*len(ACTIONS)))

    print("Début de l'apprentissage SARSA avec",N,"épisodes et lambda = ", lambda_value)
   
    for _ in range(N):
        Esa = np.zeros((N_DEALER, N_PLAYER, len(ACTIONS)))

        state = init_game()
        terminal = False

        # Choisir la première action
        action = greedy_policy(state, Qsa)

        while not terminal:
            dealer_i = state["dealer"] - 1
            player_i = state["player"] - 1
            action_i = ACTION_TO_INDEX[action]

            # Exécuter l'action
            state_next, reward, terminal = step(state, action)

            # Choisir l'action suivante
            if terminal:
                delta = reward - Qsa[dealer_i, player_i, action_i]
                action_next = None
                next_Qsa = 0.0
            else:
                action_next = greedy_policy(state_next, Qsa, Ns, N0)
                next_Qsa = Qsa[state_next["dealer"] - 1, state_next["player"] - 1, ACTION_TO_INDEX[action_next]]
                delta = reward + next_Qsa - Qsa[dealer_i, player_i, action_i]

            # Comptages (pour epsilon-greedy et alpha_t)
            Ns[dealer_i, player_i] += 1
            Nsa[dealer_i, player_i, action_i] += 1
            alpha_t = 1.0 / Nsa[dealer_i, player_i, action_i]

            # Traces d'éligibilité + mise à jour globale
            Esa[dealer_i, player_i, action_i] += 1.0
            Qsa += alpha_t * delta * Esa #Mise à jour globale de Qsa
            Esa *= lambda_value

            state = state_next
            if not terminal:
                action = action_next
        
        # Calcul de la MSE à chaque épisode
        MSE = calcul_MSE(Qsa, Qsa_MC)
        MSEs.append(MSE)

    return Qsa, Nsa, Ns, MSEs

