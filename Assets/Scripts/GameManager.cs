using UnityEngine;
using UnityEngine.UI;

[DefaultExecutionOrder(-100)]
public class GameManager : MonoBehaviour
{
    public static GameManager Instance { get; private set; }

    [SerializeField] private Ghost[] ghosts;
    [SerializeField] private Pacman pacman;
    [SerializeField] private Transform pellets;
    [SerializeField] private Text gameOverText;
    [SerializeField] private Text scoreText;
    [SerializeField] private Text livesText;

    [Header("RL")]
    [SerializeField] private ChompAgent agent;           // RL: leave empty for normal human play
    [SerializeField] private bool trainingMode = false;  // RL: true = no waiting, no death animation, no restart key
    [SerializeField] private int startingLives = 3;      // RL: use 1 to end the episode on the first death

    public int score { get; private set; } = 0;
    public int lives { get; private set; } = 3;

    private int ghostMultiplier = 1;

    private void Awake()
    {
        if (Instance != null) {
            DestroyImmediate(gameObject);
        } else {
            Instance = this;
        }
    }

    private void OnDestroy()
    {
        if (Instance == this) {
            Instance = null;
        }
    }

    private void Start()
    {
        NewGame();
    }

    private void Update()
    {
        // RL: the agent restarts the game by itself in training mode
        if (!trainingMode && lives <= 0 && Input.anyKeyDown) {
            NewGame();
        }
    }

    // RL: called by ChompAgent.OnEpisodeBegin
    public void StartEpisode()
    {
        CancelInvoke();
        ghostMultiplier = 1;
        NewGame();
    }

    private void NewGame()
    {
        SetScore(0);
        SetLives(startingLives); // RL: was SetLives(3)
        NewRound();
    }

    private void NewRound()
    {
        gameOverText.enabled = false;

        foreach (Transform pellet in pellets) {
            pellet.gameObject.SetActive(true);
        }

        ResetState();
    }

    private void ResetState()
    {
        for (int i = 0; i < ghosts.Length; i++) {
            ghosts[i].ResetState();
        }

        pacman.ResetState();
    }

    private void GameOver()
    {
        gameOverText.enabled = true;

        for (int i = 0; i < ghosts.Length; i++) {
            ghosts[i].gameObject.SetActive(false);
        }

        pacman.gameObject.SetActive(false);
    }

    private void SetLives(int lives)
    {
        this.lives = lives;
        livesText.text = "x" + lives.ToString();
    }

    private void SetScore(int score)
    {
        this.score = score;
        scoreText.text = score.ToString().PadLeft(2, '0');
    }

    public void PacmanEaten()
    {
        // RL: training mode = no death animation, no 3 second wait
        if (trainingMode)
        {
            SetLives(lives - 1);
            bool over = lives <= 0;

            if (over) {
                // Switch Pac-Man off so extra collisions cannot give a second penalty before the reset
                pacman.gameObject.SetActive(false);
            } else {
                ResetState();
            }

            if (agent != null) agent.OnPacmanCaught(over);
            return;
        }

        pacman.DeathSequence();

        SetLives(lives - 1);

        if (agent != null) agent.OnPacmanCaught(lives <= 0); // RL

        if (lives > 0) {
            Invoke(nameof(ResetState), 3f);
        } else {
            GameOver();
        }
    }

    public void GhostEaten(Ghost ghost)
    {
        int points = ghost.points * ghostMultiplier;
        SetScore(score + points);

        ghostMultiplier++;

        if (agent != null) agent.OnGhostEaten(); // RL: RL reward is separate from the game score
    }

    public void PelletEaten(Pellet pellet)
    {
        pellet.gameObject.SetActive(false);

        SetScore(score + pellet.points);

        // RL: PowerPelletEaten also calls this method, so the reward is given here only (once)
        if (agent != null) agent.OnPelletEaten(pellet is PowerPellet);

        if (!HasRemainingPellets())
        {
            pacman.gameObject.SetActive(false);

            // RL: clearing the maze ends the episode; only human play waits and starts a new round
            if (agent != null) agent.OnMazeCleared();

            if (!trainingMode) {
                Invoke(nameof(NewRound), 3f);
            }
        }
    }

    public void PowerPelletEaten(PowerPellet pellet)
    {
        for (int i = 0; i < ghosts.Length; i++) {
            ghosts[i].frightened.Enable(pellet.duration);
        }

        PelletEaten(pellet);
        CancelInvoke(nameof(ResetGhostMultiplier));
        Invoke(nameof(ResetGhostMultiplier), pellet.duration);
    }

    private bool HasRemainingPellets()
    {
        foreach (Transform pellet in pellets)
        {
            if (pellet.gameObject.activeSelf) {
                return true;
            }
        }

        return false;
    }

    private void ResetGhostMultiplier()
    {
        ghostMultiplier = 1;
    }

}