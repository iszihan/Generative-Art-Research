
void setup() {

  size(500, 500);
  noLoop();
  ellipseMode(RADIUS);
}


void draw() {

  for (int iteration = 0; iteration<76; iteration++) {
    for (int level = 9; level >= 0; level--) {
      for (int blur = 0; blur<=2; blur++) {
        // print("level:", level, "\n");
        float l = map(level, 9, 0, 1, 0.01);

        clear();
        float back_col = random(0, 255);

        float stroke_col = random(0, 255);
        while (stroke_col<back_col+50 && stroke_col > back_col-50) {
          stroke_col = random(0, 255);
        }

        background(back_col);
        for (int x = 0; x<= 10; x++) {
          noFill();
          stroke(stroke_col);
          strokeWeight(blur+1);
          float orig_x = 20;
          float orig_y = 20/l;
          ellipse(width/2, height/2, orig_x+30*x, orig_y+30*x);
        }
        PImage current = get();
        rotation(current);

        PImage temp = get(138, 138, 224, 224);
        temp.filter(BLUR,blur);
        String filename = String.format("Output/tri-r%02d-cv%03d%01d.png", level, iteration, blur);
        temp.save(filename);
      }
    }
  }
}


void rotation(PImage img) {
  float rad = random(0, TWO_PI);
  pushMatrix();
  translate(width/2, height/2);
  rotate(rad);
  translate(-width/2, -height/2);
  image(img, 0, 0);
  popMatrix();
}
