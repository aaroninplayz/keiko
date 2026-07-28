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

    // Allowed Geometric Shapes (removes cylinders, hearts, cones, etc.)
    shapeList: [
        'sphere',
        'cube',
        'torus',
        'tesseract',
        'mobius',
        'helix',
        'infinity'
    ],

    // Shape Sizes (Scale multipliers relative to base calculations)
    shapes: {
        sphere: { R: 165 },               // Sphere radius
        cube: { halfSize: 110 },          // Cube half-side length
        torus: { R_torus: 150, r_tube: 35 }, // Torus ring and tube radii
        tesseract: { scale: 170 },        // Tesseract scale factor
        mobius: { scale: 130, tube: 50 },  // Mobius strip size
        helix: { R: 80, length: 220 },     // DNA helix radius and length
        infinity: { scale: 160 }          // Infinity loop size
    }
};

// Export config if using module system, otherwise attach to window
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PARTICLE_CONFIG;
} else {
    window.PARTICLE_CONFIG = PARTICLE_CONFIG;
}
