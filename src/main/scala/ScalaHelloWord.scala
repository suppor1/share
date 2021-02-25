import java.util

import scala.Int.int2long
import scala.collection.{immutable, mutable}
import scala.collection.mutable._
import scala.collection.concurrent.TrieMap
import scala.jdk.CollectionConverters._

object ScalaHelloWord {



  def main(args: Array[String]): Unit = {
    baseList
//    baseMap
  }

  def baseMap: Unit = {
    val map = Map("a"->1, "b"->2)
    println(map)
    map("a") = 3
    val mutableMap = new mutable.HashMap[String,Int]()
    mutableMap += ("A" -> 1)
    val immutableMap = new immutable.HashMap[String,Int]()
//    immutableMap += ("A" -> 1)
//    val newi = immutableMap - "a"
    val mutableSortMap = new mutable.TreeMap[String,Int]()
    val immutableSortMap = new immutable.TreeMap[String,Int]()
    val mutableLineHashMap = new mutable.LinkedHashMap[String,Int]()
//    val immutableLinkedHashMap = new immutable.Lin

    //java scala map 互相操作
    val scalaMap:mutable.Map[String,Int] = new util.HashMap[String,Int]().asScala
//    asScala 只能转出可变的Map
//    val immutableScalaMap:immutable.Map[String,Int] = new util.HashMap[String,Int]().asScala

    val mutableConcurrentMap:scala.collection.concurrent.Map[String,Int] = new TrieMap[String,Int]()

  }

  private def baseList = {
    println("A")
    println(1.toInt)
    println(sum(5))
    System.out.println(new mutable.HashMap())

    //定义数组
    val arr = new Array[String](10)

    arr.toString
    var arrBuffer = ArrayBuffer[String]()

    var arrBuffer2 = new ArrayBuffer[String]()
    arrBuffer2 ++= Array("a", "B", "C")
    arrBuffer2.trimEnd(5)
    arrBuffer2.toString()
    arr.toBuffer
    val fun = (f: String) => {
      f + "A"
    }
    arrBuffer2.foreach(fun)

    val arr1 = Array(1, 2, -4, -5, -6, -7, -8, 9)

    val arr2 = for (i <- arr1 if i > 4) yield i + 1
    arr2.foreach(println)

    var builder = new ProcessBuilder(arrBuffer2.asJava)

    var buf: Buffer[String] = builder.command().asScala
    println(mul(2, 3))
  }

  def sum(num: Int): Int = {
    var sumNum: Int = 0
    for (i <- 1 to num; from = 4 - 1; j <- from to num) {
      printf("i :%d, j:%d\n", i, j)
      sumNum += (i + j)
    }
    sumNum
  }

  def mul(base: Int, n: Int): Int = {

    if (n >= 0) {

    } else {

    }


    if (n == 0) {
      1
    } else {
      val res = base * mul(base, n -1)
      res
    }
  }

}
