let img;
let poseNet;
let poses = [];

function setup() {
  createCanvas(640, 360);

  // crear una imagen utilizando la biblioteca p5 dom
  // llamar a modelReady() cuando se cargue
  img = createImg("deport.jpg", imageReady);
  // establecer el tamaño de la imagen al tamaño del lienzo
  img.size(width, height);

  img.hide(); // ocultar la imagen en el navegador
  frameRate(1); // establece el frameRate a 1 ya que no necesitamos que se ejecute rápidamente en este caso
}

// cuando la imagen esté lista, entonces carga poseNet
function imageReady() {
  // cuando la imagen esté lista, entonces carga poseNet
  const options = {
    minConfidence: 0.1,
    inputResolution: { width, height },
  };

  // assign poseNet
  poseNet = ml5.poseNet(modelReady, options);
  // Esto establece un evento que escucha los eventos de 'pose'
  poseNet.on("pose", function(results) {
    poses = results;
  });
}

// cuando poseNet esté lista, haga la detección
function modelReady() {
  select("#status").html("Model Loaded");

  // Cuando el modelo esté listo, ejecuta la función singlePose()...
  // Si/Cuando se detecta una pose, poseNet.on('pose', ...) estará a la escucha de los resultados de la detección
  // en el bucle draw(), si hay alguna pose, ejecuta los comandos de dibujo
  poseNet.singlePose(img);
}

// draw() no mostrará nada hasta que se encuentren las poses
function draw() {
  if (poses.length > 0) {
    image(img, 0, 0, width, height);
    drawSkeleton(poses);
    drawKeypoints(poses);
    noLoop(); // stop looping when the poses are estimated
  }
}

// The following comes from https://ml5js.org/docs/posenet-webcam
// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  // Recorrer en bucle todas las poses detectadas
  for (let i = 0; i < poses.length; i += 1) {
    // Para cada pose detectada, recorre en bucle todos los puntos clave
    const pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j += 1) {
      // Un punto clave es un objeto que describe una parte del cuerpo (como rightArm o leftShoulder)
      const keypoint = pose.keypoints[j];
      // Sólo dibuja una elipse si la probabilidad de pose es mayor que 0.2
      if (keypoint.score > 0.2) {
        fill(255);
        stroke(20);
        strokeWeight(4);
        ellipse(round(keypoint.position.x), round(keypoint.position.y), 8, 8);
      }
    }
  }
}
// Una función para dibujar los esqueletos
function drawSkeleton() {
  // Recorrer en bucle todos los esqueletos detectados
  for (let i = 0; i < poses.length; i += 1) {
    const skeleton = poses[i].skeleton;
    // Para cada esqueleto, recorre todas las conexiones del cuerpo
    for (let j = 0; j < skeleton.length; j += 1) {
      const partA = skeleton[j][0];
      const partB = skeleton[j][1];
      stroke(255);
      strokeWeight(1);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }
  }
}