package com.scala.clazz

class PropertyTest {
  var a = 0
}

object PropertyTest {
  def main(args: Array[String]): Unit = {
    val p = new PropertyTest
    println(p.a)
    p.a = 2
    println(p.a)
  }
}
