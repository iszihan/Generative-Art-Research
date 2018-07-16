float smt;


void setup() {
  size(224, 224);

  noLoop();
}

void draw() {

  for (int iteration = 0; iteration < 300; iteration++) {
    for (int level =0; level<10; level++) {

      clear();

      float x1= random(30, width-30);
      float y1= random(30, width-30);

      float x2= random(30, width-30);
      while ( abs(x2-x1)<40 ) {
        x2= random(30, width-30);
      }
      float y2= random(30, width-30);
      while ( abs(y2-y1)<40 ) {
        y2= random(30, width-30);
      }
      if (level <=6) {
        smt = map(level, 0, 3, 1, 0.8);
      } else if (level <= 9) {
        smt = map(level, 7, 9, 0.4, 0);
      }
      dropColor(x1, y1, smt);   

      PImage temp = get();
      String fileName = String.format("Output/tri-c%02d-rnoi%03d.png", level, iteration);
      temp.save(fileName);
    }
  }
}

void dropColor(float cx, float cy, float smt) {
  int gap= int(random(30,50));
  int r = 30;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float distanceFromP2 = dist(x, y, cx, cy);
      float noise = random(0, 255);
      if(distanceFromP2 <= r){
        stroke(smt*1.2*distanceFromP2-20+(1-smt)*noise);
        point(x, y);
      }
      else if (distanceFromP2 <= r+gap){
        stroke(1.2*distanceFromP2-20);
        point(x, y);
      }
      else if (distanceFromP2 <= 2*r+gap){
        stroke(smt*1.2*distanceFromP2-20+(1-smt)*noise);
        point(x, y);
      }
      else if (distanceFromP2 <= 2*r+2*gap){
        stroke(1.2*distanceFromP2-20);
        point(x, y);
      }
      else if (distanceFromP2 <= 3*r+2*gap){
        stroke(smt*1.2*distanceFromP2-20+(1-smt)*noise);
        point(x, y);
      }
      else if (distanceFromP2 <= 3*r+3*gap){
        stroke(1.2*distanceFromP2-20);
        point(x, y);
      }
      else if (distanceFromP2 <= 4*r+3*gap){
        stroke(smt*1.2*distanceFromP2-20+(1-smt)*noise);
        point(x, y);
      }
    }
  }
}
