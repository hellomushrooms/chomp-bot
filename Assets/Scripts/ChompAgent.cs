using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class ChompAgent : Agent
{
    [Header("References")]
    [SerializeField] private Pacman pacman;
    [SerializeField] private Ghost[] ghosts;
    [SerializeField] private Transform pellets;       // same parent object GameManager uses
    [SerializeField] private LayerMask obstacleLayer; // same layer as Movement.obstacleLayer

    [Header("Grid (world position of the lower-left CORNER of the maze)")]
    [SerializeField] private Vector2 origin = new Vector2(-14f, -15f);
    [SerializeField] private int width = 28;
    [SerializeField] private int height = 31;

    [Header("Rewards (Interface Contract sheet; Python can override them per run)")]
    public float pelletReward = 1f;
    public float powerPelletReward = 5f;
    public float ghostReward = 20f;
    public float winReward = 100f;
    public float deathPenalty = -50f;
    public float stepPenalty = -0.01f;

    // Channels: 0 walls, 1 pellets, 2 power pellets, 3 Pac-Man, 4 ghosts, 5 frightened ghosts
    public const int Channels = 6;
    public int ObservationSize => Channels * width * height;

    private Movement movement;
    private float[] obsBuffer;
    private bool[,] walls;
    private Pellet[] allPellets;
    private Vector2Int[] pelletCells;

    public override void Initialize()
    {
        movement = pacman.GetComponent<Movement>();
        obsBuffer = new float[ObservationSize];
        Debug.Log("ChompAgent: ObservationSize = " + ObservationSize + " (Behavior Parameters must match this)");
    }

    // ---------- Episode handling ----------

    public override void OnEpisodeBegin()
    {
        // Let Python change rewards and the random seed between runs (reward-shaping experiment, fixed evaluation seeds)
        var p = Academy.Instance.EnvironmentParameters;
        pelletReward = p.GetWithDefault("pellet_reward", pelletReward);
        powerPelletReward = p.GetWithDefault("power_pellet_reward", powerPelletReward);
        ghostReward = p.GetWithDefault("ghost_reward", ghostReward);
        winReward = p.GetWithDefault("win_reward", winReward);
        deathPenalty = p.GetWithDefault("death_penalty", deathPenalty);
        stepPenalty = p.GetWithDefault("step_penalty", stepPenalty);

        int seed = (int)p.GetWithDefault("seed", -1f);
        if (seed >= 0) Random.InitState(seed);

        GameManager.Instance.StartEpisode();
    }

    // Called by GameManager when game events happen. These are the RL rewards;
    // they are separate from the game's own score.
    public void OnPelletEaten(bool isPowerPellet) => AddReward(isPowerPellet ? powerPelletReward : pelletReward);
    public void OnGhostEaten() => AddReward(ghostReward);

    public void OnPacmanCaught(bool gameOver)
    {
        AddReward(deathPenalty);
        if (gameOver) EndEpisode();
    }

    public void OnMazeCleared()
    {
        AddReward(winReward);
        EndEpisode();
    }

    // ---------- Observations ----------

    public override void CollectObservations(VectorSensor sensor)
    {
        BuildObservation(obsBuffer);
        sensor.AddObservation(obsBuffer);
    }

    // Public so the demo mode (ONNX model run with Unity's inference package) can reuse exactly the same state.
    public void BuildObservation(float[] buf)
    {
        EnsureStatic();
        System.Array.Clear(buf, 0, buf.Length);

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                if (walls[x, y]) Set(buf, 0, x, y);

        for (int i = 0; i < allPellets.Length; i++)
        {
            if (!allPellets[i].gameObject.activeSelf) continue;
            int channel = allPellets[i] is PowerPellet ? 2 : 1;
            Set(buf, channel, pelletCells[i].x, pelletCells[i].y);
        }

        Vector2Int pc = Cell(pacman.transform.position);
        Set(buf, 3, pc.x, pc.y);

        foreach (Ghost ghost in ghosts)
        {
            if (!ghost.gameObject.activeSelf) continue;
            Vector2Int gc = Cell(ghost.transform.position);
            Set(buf, ghost.frightened.enabled ? 5 : 4, gc.x, gc.y);
        }
    }

    private void EnsureStatic()
    {
        if (walls != null) return;

        Physics2D.SyncTransforms();
        walls = new bool[width, height];
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                Vector2 centre = origin + new Vector2(x + 0.5f, y + 0.5f);
                walls[x, y] = Physics2D.OverlapBox(centre, Vector2.one * 0.4f, 0f, obstacleLayer) != null;
            }
        }

        allPellets = pellets.GetComponentsInChildren<Pellet>(true);
        pelletCells = new Vector2Int[allPellets.Length];
        for (int i = 0; i < allPellets.Length; i++)
        {
            pelletCells[i] = Cell(allPellets[i].transform.position);
        }
    }

    private Vector2Int Cell(Vector3 worldPosition)
    {
        return new Vector2Int(
            Mathf.FloorToInt(worldPosition.x - origin.x),
            Mathf.FloorToInt(worldPosition.y - origin.y));
    }

    private void Set(float[] buf, int channel, int x, int y)
    {
        if (x < 0 || x >= width || y < 0 || y >= height) return; // outside the grid (e.g., in a tunnel)
        buf[(channel * height + y) * width + x] = 1f;
    }

    // ---------- Actions: 0 Up, 1 Down, 2 Left, 3 Right, 4 Stay ----------

    public override void OnActionReceived(ActionBuffers actions)
    {
        ApplyAction(actions.DiscreteActions[0]);
        AddReward(stepPenalty);
    }

    // Public so the demo mode can call it with the model's chosen action.
    public void ApplyAction(int action)
    {
        switch (action)
        {
            case 0: movement.SetDirection(Vector2.up); break;
            case 1: movement.SetDirection(Vector2.down); break;
            case 2: movement.SetDirection(Vector2.left); break;
            case 3: movement.SetDirection(Vector2.right); break;
            default: movement.SetDirection(Vector2.zero, true); break; // stay (forced: skip the wall check)
        }
    }

    // Keyboard test: set Behavior Type to "Heuristic Only" to check the whole interface by hand.
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var d = actionsOut.DiscreteActions;
        if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow)) d[0] = 0;
        else if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow)) d[0] = 1;
        else if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow)) d[0] = 2;
        else if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow)) d[0] = 3;
        else d[0] = 4;
    }

    // Shows the current episode reward in the top-left of the Game view (debug only).
    private void OnGUI()
    {
        GUI.Label(new Rect(10, 10, 400, 25), "Reward: " + GetCumulativeReward().ToString("F2"));
    }

    // Draws the grid in the Scene view when this object is selected, to check alignment with the maze.
    // While playing it also draws the cells the agent thinks are walls (red) and pellets (green).
    private void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.yellow;
        for (int x = 0; x <= width; x++)
            Gizmos.DrawLine(origin + new Vector2(x, 0), origin + new Vector2(x, height));
        for (int y = 0; y <= height; y++)
            Gizmos.DrawLine(origin + new Vector2(0, y), origin + new Vector2(width, y));

        if (walls != null)
        {
            Gizmos.color = new Color(1f, 0f, 0f, 0.4f);
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                    if (walls[x, y])
                        Gizmos.DrawCube(origin + new Vector2(x + 0.5f, y + 0.5f), Vector3.one * 0.9f);
        }

        if (pelletCells != null)
        {
            Gizmos.color = new Color(0f, 1f, 0f, 0.6f);
            foreach (Vector2Int c in pelletCells)
                Gizmos.DrawCube(origin + new Vector2(c.x + 0.5f, c.y + 0.5f), Vector3.one * 0.3f);
        }
    }
}