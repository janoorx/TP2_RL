from ex_TP2_1 import *
from ex_TP2_2 import *
from ex_TP2_3 import *

def train_Q_learning_lambda(n_episodes, N0,Qsa_MC):
    #Lambda value = 0 pour TD(0) 
    lambda_value = 0
    #MSEs à chaqeu épisode d'apprentissage
    MSEs = []


    Qsa, Nsa, Ns = init_tables()
    print("Début de l'apprentissage Q-learning(λ) avec", n_episodes, "épisodes et lambda =", lambda_value)
   
    for _ in range(n_episodes):
        Esa = np.zeros((N_DEALER, N_PLAYER, len(ACTIONS)))

        state = init_game()
        terminal = False

        # Choisir la première action
        action = greedy_policy(state, Qsa, Ns, N0)

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
            else:
                action_next = greedy_policy(state_next, Qsa, Ns, N0)
                
                #Q-Learning : on utilise la valeur max de Q(s', a') au lieu de Q(s', a')
                #Target = reward + max_{a'} Q(s', a')
                #delta = Target - Q(s, a)
                max_next_Qsa = np.max(Qsa[state_next["dealer"] - 1, state_next["player"] - 1])
                delta = reward + max_next_Qsa - Qsa[dealer_i, player_i, action_i]

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

def plot_MSE_Q_learning_lambda(list_MSEs):
    #Il y a deux lambda (0 et 1) et le MSE du Q-learning
    for name, MSEs in list_MSEs:
        if(name == "Q-learning"):
            plt.plot(MSEs, label=name,linestyle="solid", color="red")
        else:
            plt.plot(MSEs, label=f"lambda = {name}")
    plt.xlabel('Episodes')
    plt.ylabel('MSE with Monte Carlo Optimal Qsa')
    plt.title('MSE of Q-learning(λ) and SARSA with different λ values')
    plt.legend()
    plt.grid()
    plt.show()


def ex_TP3_2():
    choice = input("\033[1;36mVoulez-vous exécuter la question 2 (MSE) ou la question 3 (surface de valeur) ou quitter l'exercice 'q'? (2, 3 ou q) : \033[0;37m")
    # Nombre d'épisodes d'apprentissage
    N = 100000
    # Paramètre de contrôle de l'exploration
    N0 = 100
    Qsa_MC, _, _ = train_monte_carlo_control(N, N0) #On récupère uniquement Qsa du Monte Carlo pour le comparer avec SARSA(λ)
    if choice == "2":
        MSEs = []
            
        lambda_test = 0
        _, _, _, MSEs_0 = train_sarsa_lambda(N,N0,lambda_test,Qsa_MC)
        MSEs.append((lambda_test, MSEs_0))

        lambda_test = 1
        _, _, _, MSEs_1 = train_sarsa_lambda(N,N0,lambda_test,Qsa_MC)
        MSEs.append((lambda_test, MSEs_1))
        
        _, _, _, MSEs_Q_learning = train_Q_learning_lambda(N,N0,Qsa_MC)
        MSEs.append(("Q-learning", MSEs_Q_learning))

        plot_MSE_Q_learning_lambda(MSEs)
    elif choice == "3":  
        Qsa_Q, _, _, _ = train_Q_learning_lambda(N,N0,Qsa_MC)
        Vs = compute_optimal_value_function(Qsa_Q)
        plot_value_surface(Vs)
    elif choice == "q":
        print("\033[1;32mVous avez quitté l'exercice 2 du TP3.\033[0;37m")
    else:
        print("\033[1;31mChoix invalide. Veuillez entrer 2 ou 3.\033[0;37m")
