int cnt = 0;
float back_col;
void setup() {
  size(224, 224);
  frameRate(99999999);
}

void randrect(float rad) {  
  float r = random(180);
  while ( r == back_col){
    r = random(180);
  }
  stroke(r);
  fill(r);
  float x = random(0, width-10);
  float y = random(0, height-10);
  float w = random(width/8, width/3);
  float h = random(height/8, height/3);
  float a = random(PI);
  float m = min(w, h) / 2;
  pushMatrix();
  translate(x, y);
  rotate(a);
  rect(0, 0, w, h, rad * m);
  popMatrix();
}

void draw() {
  for (int round = 0; round < 100; round++) {
    for (int messy = 1; messy < 101; messy++) {
      clear();
      back_col = random(0, 255);
      background(back_col);
      
      float rad = round / 99.0;
      for (int i=0; i < Math.round((messy+3)/2); i++) {
        randrect(rad);
      }
      
      cnt++;
      print(cnt + "\n");
      String fname = String.format("Output/rmr-r%02d-m%02d-%03d.png", round, messy, cnt % 1000);
      save(fname);
    }
  }
  stop();
}
