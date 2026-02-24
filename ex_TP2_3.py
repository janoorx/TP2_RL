from ex_TP2_1 import *
from ex_TP2_2 import *

# SARSA(λ) — State-Action-Reward-State-Action avec traces d'éligibilité


def train_sarsa_lambda(n_episodes, N0,lambda_value,Qsa_MC):
    
    #MSEs à chaqeu épisode d'apprentissage
    MSEs = []


    Qsa, Nsa, Ns = init_tables()
    print("Début de l'apprentissage SARSA avec",n_episodes,"épisodes et lambda = ", lambda_value)
   
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

def calcul_MSE(Qsa1, Qsa2):
    return np.mean((Qsa1 - Qsa2) ** 2)

def test_policy_variable_lambdas(N, N0, lambda_values, Qsa_MC):
    MSEs = []
    for lambda_val in lambda_values:
        Qsa_SARSA, _, _, _ = train_sarsa_lambda(N,N0,lambda_val,Qsa_MC)
        MSE = calcul_MSE(Qsa_SARSA, Qsa_MC)
        MSEs.append(MSE)
        print("MSE pour lambda =", lambda_val, ":", MSE)
    
    return MSEs

def plot_MSE_lambda(MSEs, lambda_values):
    print("Affichage graphique de  la MSE en fonction de λ ...")
    plt.figure()
    plt.plot(lambda_values, MSEs, marker='o')
    plt.xlabel('λ')
    plt.ylabel('MSE entre Qsa SARSA(λ) et Qsa Monte Carlo')
    plt.title('MSE en fonction de λ')
    plt.grid()
    plt.show()

def plot_MSEs_learning(list_MSEs):
    print("Affichage graphique de la MSE en fonction des épisodes d'apprentissage ...")
    plt.figure()
    for lambda_val, MSEs in list_MSEs:
        plt.plot(MSEs, label=f'λ={lambda_val}')
    plt.xlabel('Épisodes d\'apprentissage')
    plt.ylabel('MSE entre Qsa SARSA(λ) et Qsa Monte Carlo')
    plt.title('MSE en fonction des épisodes d\'apprentissage pour différentes valeurs de λ')
    plt.legend()
    plt.grid()
    plt.show()


def ex_TP2_3():
    # Nombre d'épisodes d'apprentissage
    N = 10000

    # Paramètre de contrôle de l'exploration
    N0 = 100
    #Nombre de lambdas 
    N_lambda = 10
    #Calcul des différentes valeurs de lambda à tester
    lambda_values = [i / N_lambda for i in range(0, N_lambda + 1)] 

    Qsa_MC, _, _ = train_monte_carlo_control(N, N0) #On récupère uniquement Qsa du Monte Carlo pour le comparer avec SARSA(λ)

    choice = input("\033[1;36mVoulez-vous :\n1. Calculer l’erreur quadratique moyenne (MSE) sur tous les états s et les actions a.\n2. Pour λ = 0 et λ = 1 uniquement, tracer la courbe d’apprentissage de la MSE en fonction du numéro d’épisode.\n'q' : quitter\nEntrez 1, 2 ou q : \033[0;37m")
    
    if (choice == '1'):
        MSEs = test_policy_variable_lambdas(N,N0,lambda_values,Qsa_MC)
        plot_MSE_lambda(MSEs, lambda_values)
    elif (choice == '2'):
        MSEs_lambda = []
        
        lambda_test = 0
        _, _, _, MSEs_0 = train_sarsa_lambda(N,N0,lambda_test,Qsa_MC)
        MSEs_lambda.append((lambda_test, MSEs_0))

        lambda_test = 1
        _, _, _, MSEs_1 = train_sarsa_lambda(N,N0,lambda_test,Qsa_MC)
        MSEs_lambda.append((lambda_test, MSEs_1))
        
        plot_MSEs_learning(MSEs_lambda)
    elif (choice == 'q'):
        print("\033[1;32mVous avez quitté l'exercice 3 du TP2.\033[0;37m")
    else:
        print("\033[1;31mChoix invalide. Veuillez entrer 1 ou 2.\033[0;37m")
