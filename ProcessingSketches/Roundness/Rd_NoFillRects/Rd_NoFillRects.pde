

void setup() {
  size(200, 200);
  noFill();
  noLoop();
}

void randrect(float rad) {

  float x = random(10, width-10);
  float y = random(10, height-10);
  float w = random(width/8, width/2);
  float h = random(w-20, w+20);
  float a = random(PI);
  float m = min(w, h) / 2;
  pushMatrix();
  translate(x, y);
  rotate(a);
  float factor = rad/9;
  rect(0, 0, w, h, factor * m);
  popMatrix();
}

void draw() {
  for (int iteration = 0; iteration < 100; iteration++) {
    for (int rad = 0; rad<10; rad++) {
      for (int blur=0; blur<=4; blur++) {

        clear();
        float back_col = random(0, 255);
        background(back_col);

        for(int i=0; i<random(2,25);i++){
          float r;
          do {
              r = random(0, 255);
            } while ( r < back_col+70 && r > back_col-70);
          stroke(r);
          float wt;
          if(blur<=2){
            wt = random(2, 10);
          }
          else{
            wt = random(5, 10);
          }
          strokeWeight(wt);

          if (rad!=9) {
            randrect(rad);
          }
          else if (rad==9) {
            float radius = random(width/8, width/2);
            float x = random(20, width-20);
            float y = random(20, height-20);
            ellipse(x, y, radius, radius);
          }
        }

        PImage temp = get();
        float b = blur;
        if (blur==2) {
          b = 1.5;
        }
        
        temp.filter(BLUR, b);
        String fname = String.format("Output/tri-r%02d-%03db%01d.png", rad, iteration, blur);
        temp.save(fname);
      }
    }
  }
}