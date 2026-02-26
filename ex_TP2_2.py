from ex_TP2_1 import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#MONTE CARLO CONTROL

ACTIONS = ("hit", "stick")
ACTION_TO_INDEX = {"hit": 0, "stick": 1}
N_DEALER = 10 
N_PLAYER = 21

def greedy_policy(state, Qsa, Ns, N0):
    """
    Politique epsilon-greedy

    Args:
        state (dict): état actuel du jeu
        Qsa (np.array): table de valeurs d'action Q(s, a)
        Ns (np.array): table du nombre de visites de chaque état Ns(s)
        N0 (int): paramètre de contrôle de l'exploration
    Returns:
        str: action choisie ("hit" ou "stick")
    
    """
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


def init_tables():
    """
    Initialise les tables Q(s, a), N(s, a) et N(s) pour le contrôle Monte Carlo

    Returns:
        Qsa (np.array): table de valeurs d'action Q(s, a) initialisée à zéro
        Nsa (np.array): table du nombre de visites de chaque paire (s, a) initialisée à zéro
        Ns (np.array): table du nombre de visites de chaque état s initialisée à zéro
    """
    Qsa = np.zeros((N_DEALER, N_PLAYER, len(ACTIONS)))
    Nsa = np.zeros((N_DEALER, N_PLAYER, len(ACTIONS)))
    Ns = np.zeros((N_DEALER, N_PLAYER))
    return Qsa, Nsa, Ns


def train_monte_carlo_control(n_episodes, N0):
    """
    Cette fonction implémente l'algorithme de contrôle Monte Carlo pour apprendre la politique optimale au Blackjack.

    Args:
        n_episodes (int): nombre d'épisodes d'apprentissage
        N0 (int): paramètre de contrôle de l'exploration pour la politique epsilon-greedy
    Returns:
        Qsa (np.array): table de valeurs d'action Q(s, a) apprise
        Nsa (np.array): table du nombre de visites de chaque paire (s, a) après l'apprentissage
        Ns (np.array): table du nombre de visites de chaque état s après l'apprentissage
    """
    Qsa, Nsa, Ns = init_tables()

    print("Début de l'apprentissage MC avec", n_episodes, "épisodes...")
    progress_step = max(1, n_episodes // 10)

    for i in range(n_episodes):
        state = init_game()
        terminal = False

        # Tableau des (state, action) rencontrés pendant l'épisode
        sa_meet = []

        while not terminal:
            action = greedy_policy(state, Qsa, Ns, N0)
            sa_meet.append((state["dealer"] - 1, state["player"] - 1, ACTION_TO_INDEX[action]))
            state, reward, terminal = step(state, action)

        # Mise à jour MC : on utilise ici la récompense terminale pour tous les (s,a) visités
        for (dealer_i, player_i, action_i) in sa_meet:
            d = dealer_i
            p = player_i
            a = action_i
            
            Nsa[d, p, a] += 1
            Ns[d, p] += 1

            alpha_t = 1.0 / Nsa[d, p, a]
            Qsa_prev = Qsa[d, p, a]
            Qsa[d, p, a] = Qsa_prev + alpha_t * (reward - Qsa_prev)

        if i % progress_step == 0:
            print("Avancement de l'apprentissage : ", (i / n_episodes) * 100, "%")
            print("[" + "\033[32m" + "#" * int((i / n_episodes) * 20) + "\033[0m" + "-" * (20 - int((i / n_episodes) * 20)) + "]")
            print("\033[F\033[K", end='')  # Efface la ligne précédente

    print("Fin de l'apprentissage.")
    return Qsa, Nsa, Ns


def evaluate_policy(Qsa, n_test):
    """
    Évalue la politique apprise à partir de la table Q(s, a) en jouant un certain nombre de parties et en calculant le taux de victoire.

    Args: 
        Qsa (np.array): table de valeurs d'action Q(s, a) apprise
        n_test (int): nombre de parties à jouer pour évaluer la politique
    Returns:
        float: taux de victoire de la politique apprise sur les parties testées
    """
    total_win = 0
    for _ in range(n_test):
        state = init_game()
        terminal = False

        while not terminal:
            dealer_i = state["dealer"] - 1
            player_i = state["player"] - 1
            action_i = Qsa[dealer_i, player_i, 1] > Qsa[dealer_i, player_i, 0]  # 1 -> stick, 0 -> hit
            action = ACTIONS[action_i]
            state, reward, terminal = step(state, action)

        if reward == 1:
            total_win += 1

    return total_win / n_test


def compute_optimal_value_function(Qsa):
    """
    Calcule la fonction de valeur optimale V*(s) à partir de la table Q(s, a) apprise.

    Args:
        Qsa (np.array): table de valeurs d'action Q(s, a) apprise
    Returns:
        np.array: table de la fonction de valeur optimale V*(s) pour chaque état s
    """
    return Qsa.max(axis=2)


def plot_value_surface(Vs):
    """
    Affiche graphiquement la fonction de valeur optimale V*(s) en 3D, avec l'axe X représentant la valeur du dealer, 
    l'axe Y représentant la valeur du joueur, et l'axe Z représentant V*(s).

    Args:
        Vs (np.array): table de la fonction de valeur optimale V*(s) pour chaque état s
    """
    print("Affichage graphique de la politique optimale...")
    X = np.arange(1, N_DEALER + 1)
    Y = np.arange(1, N_PLAYER + 1)
    X, Y = np.meshgrid(X, Y)
    Z = np.asarray(Vs).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Value of Dealer')
    ax.set_ylabel('Value of Player')
    ax.set_zlabel('Optimal Value Function V*(s)')
    ax.set_title('Optimal Value Function V*(s)')
    plt.show()

def ex_TP2_2():
    """
    Execute l'algorithme de contrôle Monte Carlo pour apprendre la politique optimale au Blackjack, 
    évalue la politique apprise en jouant un certain nombre de parties, et affiche graphiquement la fonction de valeur optimale V*(s) en 3D.
    """
     # Nombre d'itérations par épisode
    N = 100000
    # Paramètre de contrôle de l'exploration
    N0 = 100
    # Nombre d'itérations du test de la politique optimale
    N_test = 10000

    Qsa, Nsa, Ns = train_monte_carlo_control(
        n_episodes=N,
        N0=N0
    )

    win_rate = evaluate_policy(Qsa, N_test)
    print("Win rate sur", N_test, "parties : ", win_rate * 100, "%")

    Vs = compute_optimal_value_function(Qsa)
    plot_value_surface(Vs)

   


