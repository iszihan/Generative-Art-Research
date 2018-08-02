
final int ANT_ITERATIONS = 700;
final int NUM_ANTS = 300;
ArrayList<Ant> ants = new ArrayList<Ant>(NUM_ANTS);
int count = 0;

void setup() {
  size(500, 500);

  for (int i =0; i <NUM_ANTS; i++) {
    ants.add(new Ant());
  }

  background(0);
  stroke(255, 10);
  noLoop();
}


void draw() {
  for (int iteration = 0; iteration < 166; iteration++) {
    for (int i = 0; i < 10; i+=1) {
      for (int blur = 0; blur<=2; blur+=1){
        count++;
        float level = 10*i;
        background(0);
        noiseSeed(System.currentTimeMillis());
        noiseDetail(6, level/100.0 + .001); // we need the .001 to mkae sure this doesn't go to zero
        for (Ant ant : ants) {
          for (int z = 0; z < ANT_ITERATIONS; z++) {
            ant.draw(.012);
          }
        }
        
        PImage temp = get();
        temp.filter(BLUR, blur);
        
        String fileName = String.format("Output/tri-r%02d-noi%03d%01d.png", 10 - i -1, iteration, blur);
        temp.save(fileName);
        //println(String.format("%02.02f", (iteration*100 + i)/(100.0 * ITERATIONS)));
        println(count);
      }
    }
  }
}




class Ant {
  float x;
  float y;
  float heading;

  Ant() {
    randomPosition();
  }

  void draw(float noiseScale) {
    float newX, newY;

    heading = noise(x*noiseScale, y*noiseScale)*TWO_PI;

    newX = x + sin(heading);
    newY = y +  cos(heading);
    line(x, y, newX, newY);

    x = newX;
    y = newY;

    if (x <0 | x > width) {
      randomPosition();
    }
    if (y <0 | y > height) {
      randomPosition();
    }
  }

  void randomPosition() {
    x = random(0, width);
    y = random(0, height);
  }
}
