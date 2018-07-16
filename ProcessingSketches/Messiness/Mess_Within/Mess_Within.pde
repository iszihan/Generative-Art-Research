
int size;
final float mess = 2;

void setup() {
  size(224, 224);
  noLoop();
  rectMode(CENTER);
  ellipseMode(CENTER);
}


void draw() {


  for (int iteration = 0; iteration < 125; iteration++) {
    for (int m=1; m<=10; m++) {
      for (int blur=0; blur<=3; blur++) {
        
        float back_col = random(70, 240);
        float fill_col = random(50, 255);
        
        size = int(random(width/2-30, width/2+30));
        background(back_col);
        noStroke();
        fill(fill_col);
        rect(width/2, height/2, size, size);

        int level = (m-1)*(m-1);
        int scale = 10 + 40 * (level/100);
        for (int i = 0; i < (level-1) * mess + 2; i++) {  // Each primitive drawm

          float x = random(width/2-size/2+10, width/2+size/2-10);
          float y = random(height/2-size/2+10, height/2+size/2-10);
          float otherX = random(5, min(x-(width-size)/2, (width-size)/2+size-x));
          while (otherX > x-10 && otherX < x+10) {
            otherX = random(5, min(x-(width-size)/2, (width-size)/2+size-x));
          }
          float otherY = random(5, min(y-(height-size)/2, (height-size)/2+size-y));
          while (otherY > y-10 && otherY < y+10) {
            otherY = random(5, min(y-(height-size)/2, (height-size)/2+size-y));
          }
          float rotation = random(0, TWO_PI);

          if (level > 20 && random(0, 1) < 0.2) {
            noFill();
          } else {
            float col = random(50, 255);
            while (col > back_col-15 && col < back_col+15 ) {
              col = random(50, 255);
            }
            fill(col);
          }
          stroke(random(0, 150));

          pushMatrix();
          rotate((level/100) * random(-PI, PI));
          strokeWeight(1);
          int choice = (int)random(1, 6);  // Choice of which 2D primitive to draw

          switch(choice) {
          case 1:  // Arc
            float start_rot = random(0, TWO_PI);
            while (start_rot == rotation) {
              start_rot = random(0, TWO_PI);
            }
            arc(x, y, otherX, otherY, start_rot, rotation);
            break;
          case 2:  // Ellipse
            ellipse(x, y, otherX, otherY);
            break;
          case 3:  // Line
            line(x, y, x+otherX, y+otherY);
            break;
          case 4:  // Quad
            quad(x, y, x+otherX, y+otherY, x-otherX, y-otherY, x+0.5*otherX, y-0.8*otherY);
            break;
          case 5:  // Rect
            rect(x, y, random(15, scale*1.5), random(15, scale*1.5), random(0, 15));
            break;
          case 6: // Triangle
            triangle(x, y, x+otherX, y+otherY, x-otherX, y-otherY);
            break;
          default:
            print("Uh oh");
          }
          popMatrix();
          strokeWeight(2);
          stroke(random(0, 50), 20);
          for (int point = 0; point < level; point++) {
            point(random(0, width), random(0, height));
          }
        }

        PImage temp = get();
        temp.filter(BLUR, blur);
        temp.save(String.format("Output/tri-m%02d-mesw%03d%01d.png", m, iteration, blur));
        clear();
      }
    }
  }
}
