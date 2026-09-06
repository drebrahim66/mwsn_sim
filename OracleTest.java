import java.util.*;
import java.util.concurrent.*;

/** Oracle run of the ver7 Java engine on the paper's default configuration:
 *  500 legitimate nodes + 4 replicas (one per victim), 500x500 m, radius ~32 m
 *  (avg 6 neighbours), speed 3 m/s, cThresh=1, no faults, TTL=inf. */
public class OracleTest implements ThreadListener {
    List<Integer> times = Collections.synchronizedList(new ArrayList<>());
    public static void main(String[] a) throws Exception {
        int iters = a.length > 0 ? Integer.parseInt(a[0]) : 10;
        int nodes = a.length > 1 ? Integer.parseInt(a[1]) : 500;
        int imposters = a.length > 2 ? Integer.parseInt(a[2]) : 4;
        int area = a.length > 3 ? Integer.parseInt(a[3]) : 500;
        int radius = a.length > 4 ? Integer.parseInt(a[4]) : 32;
        int speed = a.length > 5 ? Integer.parseInt(a[5]) : 3;
        int simTime = 20000;
        OracleTest o = new OracleTest();
        ThreadPoolExecutor ex = new ThreadPoolExecutor(8, 8, 9, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
        int[] fp = new int[imposters]; Arrays.fill(fp, 100);
        for (int s = 0; s < iters; s++) {
            Simulation sim = new Simulation("T" + s, nodes + imposters, area, area, speed, 0, radius, 3,
                simTime, simTime, imposters, 1000, 1, false, false, false,
                Node.increasingType.incremental, simTime, Simulation.faultType.noFault, 0, 0, simTime,
                Simulation.faultClass.modification, simTime, fp, 1);
            sim.addListener(o);
            ex.submit(sim);
        }
        ex.shutdown(); ex.awaitTermination(2, TimeUnit.HOURS);
        double m = 0; for (int t : o.times) m += t; m /= Math.max(1, o.times.size());
        System.out.println("runs=" + o.times.size() + " T_q values=" + o.times + " mean=" + m);
    }
    public void onNewData(HashMap<String, Object> d) {
        times.add((Integer) d.get("DETECTION"));
    }
}
