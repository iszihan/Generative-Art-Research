final int ITERATIONS = 5;
final int ANT_ITERATIONS = 700;
final int NUM_ANTS = 300;
ArrayList<Ant> ants = new ArrayList<Ant>(NUM_ANTS);
int count = 0;
PImage img;
PGraphics mask_hori, mask_vert;

void setup() {
  size(224, 224);

  for (int i =0; i <NUM_ANTS; i++) {
    ants.add(new Ant());
  }
  
  background(0);
  stroke(255, 10);
  noLoop();
}


void draw() {
  for (int iteration = 0; iteration < ITERATIONS; iteration++) {
    for (int i = 0; i <= 100; i+=1) {
      count++;
      background(0);
      noiseSeed(System.currentTimeMillis());
      noiseDetail(6, i/100.0 + .001); // we need the .001 to mkae sure this doesn't go to zero
      for (Ant ant : ants) {
        for (int z = 0; z < ANT_ITERATIONS; z++) {
          ant.draw(.012);
        }
      }
      
      img = get();
      mask_hori = createGraphics(224,224);
      mask_hori.beginDraw();
      mask_hori.rect(0, 0, img.width/2, img.height);
      mask_hori.endDraw(); 
  
      mask_vert = createGraphics(224,224);
      mask_vert.beginDraw();
      mask_vert.rect(0, 0, img.width, img.height/2);
      mask_vert.endDraw(); 
      imageFlipped_hori(img, 0, 0 );  
      img.mask(mask_hori);
      image(img,0,0);
      String fileName = String.format("Noise-Hori/img-r%02d-s100-%03d-hori.png", 100 - i, iteration);
      save(fileName);
      
    }
  }
}

void imageFlipped_vert( PImage img, float x, float y ){
  pushMatrix(); 
  translate(0, img.height);
  scale( 1, -1 );
  image(img, x, y); 
  popMatrix(); 
} 

void imageFlipped_hori( PImage img, float x, float y ){
  pushMatrix(); 
  translate(img.width, 0);
  scale( -1, 1 );
  image(img, x, y); 
  popMatrix(); 
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
