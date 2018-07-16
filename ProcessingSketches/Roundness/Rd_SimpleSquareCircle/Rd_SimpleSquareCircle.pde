int count = 0;

void setup() {
  size(200, 200);
  //frameRate(4);
  noStroke();
}


void draw() {
  clear();
  background(random(0, 204));
  
  int roundness = 0;
  if (random(1) > 0.5) roundness = 99;
  
  for (int i = 0; i < random(2,6); i++) {
    int x = (int)random(width * 0.1, width * 0.9);
    int y = (int)random(height * 0.1, height * 0.9);
    
    float size = random(0.1 * (height + width)/2, 0.3 * (height + width)/2);
    float size2 = random(0.1 * (height + width)/2, 0.3 * (height + width)/2);
    float scaleFactor = random(-0.4, 0.4);
    
    if (scaleFactor < 0) size *= (-scaleFactor + 1);
    if (scaleFactor > 0) size2 *= (scaleFactor + 1);
    
    fill(random(48, 255));
    if (roundness == 0) rect(x, y, size, size2);
    else ellipse(x, y, size, size2);
  }
  
  for (int blur =1 ;blur<=5;blur++){
    PImage temp = get();
    temp.filter(BLUR, blur);
    String fname = String.format("SimpleRoundRects/tri-r%02d-ssc%03d%01d.png", roundness, count,blur);
    temp.save(fname);
    
  }
  
  count++;
  if (count % 100 == 0) println(count);
  if (count > 200) stop();
}
