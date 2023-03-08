// Initialize Three.js scene, camera, and renderer
var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
var renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Create a cube in the scene
var geometry = new THREE.BoxGeometry();
var material = new THREE.MeshBasicMaterial({color: 0x00fff0});
var cube = new THREE.Mesh(geometry, material);
scene.add(cube);

// Position the camera to view the cube
camera.position.z = 5;

// Animate the scene
function animate() {
  requestAnimationFrame(animate);
  cube.rotation.x += 0.01;
  cube.rotation.y += 0.01;
  renderer.render(scene, camera);
}
animate();