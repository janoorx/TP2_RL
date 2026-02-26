from ex_TP2_1 import *
from ex_TP2_2 import *

# SARSA(λ) — State-Action-Reward-State-Action avec traces d'éligibilité


def train_sarsa_lambda(n_episodes, N0,lambda_value,Qsa_MC):
    """
    Cette fonction implémente l'algorithme SARSA(λ) pour apprendre la politique optimale au Blackjack, 
    en utilisant des traces d'éligibilité pour accélérer l'apprentissage.

    Args:
        n_episodes (int): nombre d'épisodes d'apprentissage
        N0 (int): paramètre de contrôle de l'exploration pour la politique epsilon-greedy
        lambda_value (float): paramètre pour les traces d'éligibilité (0 ≤ λ ≤ 1)
        Qsa_MC (np.array): table de valeurs d'action Q(s, a) apprise par l'algorithme de contrôle Monte Carlo
    Returns:
        Qsa (np.array): table de valeurs d'action Q(s, a) apprise par SARSA(λ)
        Nsa (np.array): table du nombre de visites de chaque paire (s, a) après l'apprentissage
        Ns (np.array): table du nombre de visites de chaque état s après l'apprentissage
        MSEs (list): liste des erreurs quadratiques moyennes (MSE) entre Qsa de SARSA(λ) et Qsa_MC à chaque épisode d'apprentissage
    """
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
    """
    Calcul l'erreur quadratique moyenne (MSE) entre deux tables de valeurs d'action Q(s, a).

    Args:
        Qsa1 (np.array): première table de valeurs d'action Q(s, a)
        Qsa2 (np.array): deuxième table de valeurs d'action Q(s, a)
    Returns:
        float: l'erreur quadratique moyenne (MSE) entre Qsa1 et Qsa2
    """
    return np.mean((Qsa1 - Qsa2) ** 2)

def test_policy_variable_lambdas(N, N0, lambda_values, Qsa_MC):
    """
    Test la politique apprise par SARSA(λ) pour différentes valeurs de λ, 
    en calculant l'erreur quadratique moyenne (MSE) entre la table Q(s, a) apprise par SARSA(λ) et la table Q(s, a) apprise par Monte Carlo.

    Args:
        N (int): nombre d'épisodes d'apprentissage
        N0 (int): paramètre de contrôle de l'exploration pour la politique epsilon-greedy
        lambda_values (list): liste des différentes valeurs de λ à tester
        Qsa_MC (np.array): table de valeurs d'action Q(s, a) apprise par l'algorithme de contrôle Monte Carlo
    Returns:
        MSEs: liste des erreurs quadratiques moyennes (MSE) pour chaque valeur de λ testée
    """
    MSEs = []
    for lambda_val in lambda_values:
        Qsa_SARSA, _, _, _ = train_sarsa_lambda(N,N0,lambda_val,Qsa_MC)
        MSE = calcul_MSE(Qsa_SARSA, Qsa_MC)
        MSEs.append(MSE)
        print("MSE pour lambda =", lambda_val, ":", MSE)
    
    return MSEs

def plot_MSE_lambda(MSEs, lambda_values):
    """
    Affiche graphiquement l'erreur quadratique moyenne (MSE) entre la table Q(s, a) apprise par SARSA(λ) 
    et la table Q(s, a) apprise par Monte Carlo en fonction des différentes valeurs de λ testées.

    Args:
        MSEs (list): liste des erreurs quadratiques moyennes (MSE) pour chaque valeur de λ testée
        lambda_values (list): liste des différentes valeurs de λ
    """
    print("Affichage graphique de  la MSE en fonction de λ ...")
    plt.figure()
    plt.plot(lambda_values, MSEs, marker='o')
    plt.xlabel('λ')
    plt.ylabel('MSE entre Qsa SARSA(λ) et Qsa Monte Carlo')
    plt.title('MSE en fonction de λ')
    plt.grid()
    plt.show()

def plot_MSE(list_MSEs):
    """
    Affiche graphiquement l'erreur quadratique moyenne (MSE) entre la table Q(s, a) apprise par SARSA(λ)
    et la table Q(s, a) apprise par Monte Carlo en fonction du numéro d’épisode d’apprentissage pour différentes valeurs de λ.
   
    Args: list_MSEs (list): liste de tuples (lambda_value, MSEs), où lambda_value est la valeur de λ testée et MSEs est la liste des MSE à chaque épisode d’apprentissage pour cette valeur de λ.
    Returns: None
    """
    #!CHANGER LES DOCS STRING
    # Il y a deux lambda (0 et 1) et le MSE du Q-learning
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

def ex_TP2_3():
    """
    Fonction principale pour exécuter l'exercice 3 du TP2, qui permet à l'utilisateur de choisir entre
    calculer la MSE pour différentes valeurs de λ ou tracer la courbe d'apprentissage de la MSE pour λ=0 et λ=1.
    """
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
        
        plot_MSE(MSEs_lambda)
    elif (choice == 'q'):
        print("\033[1;32mVous avez quitté l'exercice 3 du TP2.\033[0;37m")
    else:
        print("\033[1;31mChoix invalide. Veuillez entrer 1 ou 2.\033[0;37m")
