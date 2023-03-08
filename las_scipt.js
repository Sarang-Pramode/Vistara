// Load the LAS file using LASLoader
LASLoader.load('assets/LiDAR/lasFile_Reconstructed_25192.las', function (data) {
    // Create a Three.js scene
    const scene = new THREE.Scene();
    
    // Create a Three.js point cloud with the LAS data
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(data.position, 3));
    const material = new THREE.PointsMaterial({ color: 0xffffff, size: 0.01 });
    const pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);
    
    // Create a Three.js camera
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(500, 500, 500);
    
    // Create a Three.js renderer and add it to the page
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    
    // Create an LOD object to optimize rendering performance
    const lod = new THREE.LOD();
    
    // Add the point cloud to the LOD object with multiple levels of detail
    const levelsOfDetail = [
        { distance: 0, points: pointCloud },
        { distance: 500, points: pointCloud.clone() },
        { distance: 1000, points: pointCloud.clone() }
    ];
    for (let i = 0; i < levelsOfDetail.length; i++) {
        const level = levelsOfDetail[i];
        level.points.material = material.clone();
        level.points.material.size = level.distance / 1000;
        lod.add(level.points, level.distance);
    }
    scene.add(lod);
    
    // Animate the point cloud by rotating the camera
    function animate() {
        requestAnimationFrame(animate);
        camera.position.x = Math.cos(Date.now() * 0.001) * 500;
        camera.position.z = Math.sin(Date.now() * 0.001) * 500;
        camera.lookAt(scene.position);
        renderer.render(scene, camera);
    }
    animate();
});