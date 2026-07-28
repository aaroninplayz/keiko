/**
 * KEIKO Particle System Configuration File
 * Feel free to customize any of these values to tweak the visual aesthetics,
 * shape sizes, rendering speeds, and mouse behaviors.
 */
const PARTICLE_CONFIG = {
    // Basic settings
    maxParticles: 600,                    // Total number of particle nodes
    particleRadius: 3.0,                  // Radius of each dot in pixels
    particleColor: 'rgba(210, 187, 255, 0.55)', // Color of particle dots

    // Wireframe Mesh Settings
    connectionColor: 'rgba(210, 187, 255, 0.50)', // Color of connection lines (0.50 opacity)
    connectionMaxDistance: 100,           // Max distance (px) between nodes to draw lines
    connectionMaxCount: 3,                // Max lines per particle (3 connections per node)

    // Mouse Interaction Settings
    mouseRepelDistance: 130,              // Distance (px) at which the cursor pushes particles
    mouseRepelForce: 2.5,                 // Acceleration multiplier for mouse repulsion

    // Animation Time States (in frames at 60 FPS)
    wanderDuration: 450,                  // How long particles stay in wandering "blanket" state (7.5 seconds)
    transitionDuration: 180,             // How long the morphing animation takes (3 seconds)
    shapeDuration: 600,                   // How long a shape stays formed on screen (10.0 seconds)

    // Allowed Star Systems & Constellations
    shapeList: [
        'orion',          // Orion (The Hunter & Belt)
        'big_dipper',     // Ursa Major / Big Dipper
        'cassiopeia',     // Cassiopeia (Celestial W)
        'scorpius',       // Scorpius (Scorpion & Antares)
        'cygnus',         // Cygnus (Northern Cross Swan)
        'pleiades',       // Pleiades (Seven Sisters Star Cluster)
        'galaxy_spiral'   // Galactic Spiral Star System
    ],

    // Constellation Scale Multipliers
    shapes: {
        orion: { scale: 190 },
        big_dipper: { scale: 185 },
        cassiopeia: { scale: 175 },
        scorpius: { scale: 180 },
        cygnus: { scale: 195 },
        pleiades: { scale: 170 },
        galaxy_spiral: { scale: 200 }
    }
};

// Export config if using module system, otherwise attach to window
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PARTICLE_CONFIG;
} else {
    window.PARTICLE_CONFIG = PARTICLE_CONFIG;
}
