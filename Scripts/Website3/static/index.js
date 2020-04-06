var wmes = 0;
var sensor_list = 0;
var focus = 0;
var group = 0;
var type = 0;

window.onload = function () {

    var forms = document.getElementsByClassName('needs-validation');
    var validation = Array.prototype.filter.call(forms, function (form) {
        form.addEventListener('submit', function (event) {
            perform_manual_validation();
            if (form.checkValidity() === false) {
                event.preventDefault();
                event.stopPropagation();
                form.classList.add('was-validated');
                $('#collapse-target').collapse('show');
                $('#collapse-analysis').collapse('show');
            } else {
                submit();
                form.classList.remove('was-validated');    
            };
        }, false);
    });

    add_event_listener_collapsers("target", false);
    add_event_listener_collapsers("analysis", false);
    add_event_listener_collapsers("visualization-settings", false);
    add_event_listener_collapsers("line_chart", true);
    add_event_listener_collapsers("heat_map", true);

    document.getElementById("btn-redo-analysis-settings").addEventListener("click", function(){ 
        init_form($('#source').selectpicker('val'));
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            perform_manual_validation();
            form.checkValidity();
        };
        $('#collapse-target').collapse('show');
        $('#collapse-analysis').collapse('show');
    });

    document.getElementById("btn-collapse-visualization").addEventListener("click", function(){  
        if($('#btn-collapse-visualization').prop('name') == "not-active"){
            compress_expand_visualization("compress");
        } else {
            compress_expand_visualization("expand");
        };
    });

    document.getElementById("source").addEventListener("change", function () {
        update_form(this.value);
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            perform_manual_validation();
            form.checkValidity();
        };
    });

    document.querySelectorAll("input[name=check-focus]").forEach(box => {
        box.addEventListener("change", function () {
            if (document.getElementById("form-target").classList.contains('was-validated') === true) {
                validate_check_box("focus", 1);
            };
            update_sensors(false, false);
        });
    });

    document.querySelectorAll("input[name=check-sensor-group]").forEach(box => {
        box.addEventListener("change", function () {
            if (document.getElementById("form-target").classList.contains('was-validated') === true) {
                validate_check_box("sensor-group", 1);
            };
            update_sensors(false, false);
        });
    });

    document.getElementById("sensor-type").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("sensor-type", 1);
        };
        update_sensors(false, false);
    });

    document.getElementById("sensor-name").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("sensor-name", 2);
        };
    });

    document.getElementById("calendar").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("calendar", 1);
        };
    });

    document.getElementById("correlation-type").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("correlation-type", 1);
        };
        update_correlation();
    });

    document.getElementById("correlation").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("correlation", 1);
        };
    });

    get_init_json();
};

function add_event_listener_collapsers(name, is_chart){
    $('#collapse-' + name).on('hidden.bs.collapse', function () {
        $("#btn-collapse-" + name).html('<i class="fa fa-chevron-down"></i>');
        document.getElementById('btn-collapse-' + name).title = 'Show';  
        if (is_chart){
            Plotly.Plots.resize(name);
        };
    });
    $('#collapse-' + name).on('shown.bs.collapse', function () {
        $("#btn-collapse-" + name).html('<i class="fa fa-chevron-up"></i>');
        document.getElementById('btn-collapse-' + name).title = 'Hide';
        if (is_chart){
            Plotly.Plots.resize(name);
        };
    });
};

function compress_expand_visualization(type){
    if (type == "compress") {
        $('#btn-collapse-visualization').prop('name', 'active');
        document.getElementById('row-form').className="col-sm-4";
        document.getElementById('row-visualization').className="col-sm-8";
        document.getElementById('btn-collapse-visualization').title = 'Expand';
        $("#btn-collapse-visualization").html('<i class="fa fa-expand"></i>');
    } else {
        $('#btn-collapse-visualization').prop('name', "not-active");
        document.getElementById('row-form').className="col-sm-2";
        document.getElementById('row-visualization').className="col-sm-10";
        document.getElementById('btn-collapse-visualization').title = 'Compress';
        $("#btn-collapse-visualization").html('<i class="fa fa-compress"></i>');
    };
    Plotly.Plots.resize("line_chart");
    Plotly.Plots.resize("heat_map");
};

function perform_manual_validation() {
    validate_check_box("focus", 1);
    validate_check_box("sensor-group", 1);
    validate_selection("sensor-type", 1);
    validate_selection("sensor-name", 2);
    validate_selection("calendar", 1);
    validate_selection("correlation-type", 1);
    validate_selection("correlation", 1);
};

function get_init_json() {
    console.log("get_init_json");
    var data = {
        request_type: "json"
    };
    $.post("receiver", data, receive_init_json);
};

function receive_init_json(data, status) {
    var data_json = JSON.parse(data);
    wmes = {
        infraquinta: data_json['infraquinta'],
        barreiro: data_json['barreiro'],
        beja: data_json['beja']
    };
    init_form("infraquinta");
};


function update_form(wme) {

    $('#source').selectpicker('val', wme);

    switch (wme) {
        case "infraquinta":
            sensor_list = wmes.infraquinta;
            break;
        case "barreiro":
            sensor_list = wmes.barreiro;
            break;
        case "beja":
            sensor_list = wmes.beja;
            break;
    }

    focus = {
        real: false,
        simulated: false
    };
    group = {
        telemanagement: false,
        telemetry: false
    };
    type = {
        flow: false,
        pressure: false
    };

    console.log(sensor_list)

    for (var key in sensor_list) {
        if (!sensor_list.hasOwnProperty(key)) {
            continue;
        };

        var sensor = sensor_list[key];

        switch (sensor["focus"]) {
            case "real":
                focus.real = true;
                break;
            case "simulated":
                focus.simulated = true;
                break;
        };

        switch (sensor["group"]) {
            case "telemanagement":
                group.telemanagement = true;
                break;
            case "telemetry":
                group.telemetry = true;
                break;
        };

        switch (sensor["type"]) {
            case "flow":
                type.flow = true;
                break;
            case "pressure":
                type.pressure = true;
                break;
        };
    };

    update_selections_checkboxes();
    update_sensors(true, true);
}

function update_selections_checkboxes() {

    $("[value='real'][name='check-focus']").prop("checked", false);
    if (focus.real == false) {
        $("[value='real'][name='check-focus']").prop("disabled", true);
    } else {
        $("[value='real'][name='check-focus']").prop("disabled", false);
        $("[value='real'][name='check-focus']").prop("checked", true);
    }

    $("[value='simulated'][name='check-focus']").prop("checked", false);
    if (focus.simulated == false) {
        $("[value='simulated'][name='check-focus']").prop("disabled", true);
    } else {
        $("[value='simulated'][name='check-focus']").prop("disabled", false);
        $("[value='simulated'][name='check-focus']").prop("checked", true);
    }

    $("[value='telemetry'][name='check-sensor-group']").prop("checked", false);
    if (group.telemetry == false) {
        $("[value='telemetry'][name='check-sensor-group']").prop("disabled", true);
    } else {
        $("[value='telemetry'][name='check-sensor-group']").prop("disabled", false);
        $("[value='telemetry'][name='check-sensor-group']").prop("checked", true);
    }

    $("[value='telemanagement'][name='check-sensor-group']").prop("checked", false);
    if (group.telemanagement == false) {
        $("[value='telemanagement'][name='check-sensor-group']").prop("disabled", true);
    } else {
        $("[value='telemanagement'][name='check-sensor-group']").prop("disabled", false);
        $("[value='telemanagement'][name='check-sensor-group']").prop("checked", true);
    }

    $('#sensor-type').selectpicker('deselectAll');
    if (type.flow == false) {
        $('#sensor-type option[value=flow]').prop("disabled", true);
    } else {
        $('#sensor-type option[value=flow]').prop("disabled", false);
        $('#sensor-type option[value=flow]').prop("selected", true);
    }
    if (type.pressure == false) {
        $('#sensor-type option[value=pressure]').prop("disabled", true);
    } else {
        $('#sensor-type option[value=pressure]').prop("disabled", false);
        $('#sensor-type option[value=pressure]').prop("selected", true);
    }
    $('#sensor-type').selectpicker('refresh');
    $('#sensor-type').selectpicker('render');
}

function update_sensors(delete_sensors, select_all) {

    if (delete_sensors) {
        $('#sensor-name').empty();
        $('#sensor-name').selectpicker('refresh');
        $('#sensor-name').selectpicker('render');
    }

    for (var key in sensor_list) {
        if (!sensor_list.hasOwnProperty(key)) {
            continue;
        };

        var sensor = sensor_list[key];
        var approved = true;

        switch (sensor["focus"]) {
            case "real":
                if ($("[name='check-focus'][value='real']:checked").length == 0) {
                    approved = false;
                }
                break;
            case "simulated":
                if ($("[name='check-focus'][value='simulated']:checked").length == 0) {
                    approved = false;
                }
                break;
        };

        switch (sensor["group"]) {
            case "telemanagement":
                if ($("[name='check-sensor-group'][value='telemanagement']:checked").length == 0) {
                    approved = false;
                }
                break;
            case "telemetry":
                if ($("[name='check-sensor-group'][value='telemetry']:checked").length == 0) {
                    approved = false;
                }
                break;
        };

        switch (sensor["type"]) {
            case "flow":
                if ($("#sensor-type option[value=flow]:selected").length == 0) {
                    approved = false;
                }
                break;
            case "pressure":
                if ($("#sensor-type option[value=pressure]:selected").length == 0) {
                    approved = false;
                }
                break;
        };

        if (approved && ($("#sensor-name option[value='" + key + "']").length == 0)) {
            var data_tokens = sensor["focus"] + " " + sensor["group"] + " " + sensor["type"];
            var data_subtext = dim(sensor["focus"]) + " • " + dim(sensor["group"]) + " • " + dim(sensor["type"]);
            var option = '<option value="' + key + '" data-subtext="' + data_subtext + '" data-tokens="' + data_tokens + '">' + sensor["name"] + '</option>';
            $('#sensor-name').append(option);
        } else {
            if ((!approved) && ($("#sensor-name option[value='" + key + "']").length != 0)) {
                $('#sensor-name').find('[value=' + key + ']').remove();
            }
        }
    };

    $('#sensor-name').selectpicker('refresh');
    $('#sensor-name').selectpicker('render');

    if (select_all) {
        $('#sensor-name').selectpicker('selectAll');
    }
}

function dim(name) {
    switch (name) {
        case "real":
            return "real";
        case "simulated":
            return "simu";
        case "telemanagement":
            return "tlmg";
        case "telemetry":
            return "tlmt";
        case "flow":
            return "flow";
        case "pressure":
            return "press";
    }
}

function update_correlation() {

    selection = $('#correlation-type').val();
    correlation_temporal = [
        ["dcca", "DCCA"],
        ["dcca-ln", "DCCA-ln"]
    ];
    correlation_tuples = [
        ["pearson", "Pearson"],
        ["kullback-leibler", "Kullback-Leibler"]
    ];

    if (selection.includes("tuples")) {
        correlation_tuples.forEach(correlation => {
            if (($("#correlation option[value='" + correlation[0] + "']").length == 0)) {
                option = '<option data-subtext="Tuples" data-tokens="Tuples" value="' + correlation[0] + '">' + correlation[1] + '</option>';
                $('#correlation').append(option);
            };
        });
    } else {
        correlation_tuples.forEach(correlation => {
            if (($("#correlation option[value='" + correlation[0] + "']").length > 0)) {
                $('#correlation').find('[value=' + correlation[0] + ']').remove();
            };
        });
    }

    if (selection.includes("temporal")) {
        correlation_temporal.forEach(correlation => {
            if (($("#correlation option[value='" + correlation[0] + "']").length == 0)) {
                option = '<option data-subtext="Temporal" data-tokens="Temporal" value="' + correlation[0] + '">' + correlation[1] + '</option>';
                $('#correlation').append(option);
            };
        });
    } else {
        correlation_temporal.forEach(correlation => {
            if (($("#correlation option[value='" + correlation[0] + "']").length > 0)) {
                $('#correlation').find('[value=' + correlation[0] + ']').remove();
            };
        });
    }

    $('#correlation').selectpicker('refresh');
    $('#correlation').selectpicker('render');
}


function submit() {
    
    deactivate_form();

    $('#collapse-target').collapse('hide');
    $('#collapse-analysis').collapse('hide');
    $('#collapse-line_chart-section').collapse('hide');
    $('#collapse-heat_map-section').collapse('hide');

    window.scroll({
        top: 0, 
        left: 0, 
        behavior: 'smooth'
    });

    var data = {
        request_type: "form",  
        wme: $('#source').selectpicker('val'),
        sensors_id: JSON.stringify($('#sensor-name').selectpicker('val')),
        date_range_min: $('#date-range').data('daterangepicker').startDate.format('YYYY-MM-DD'),
        date_range_max: $('#date-range').data('daterangepicker').endDate.format('YYYY-MM-DD'),
        calendar: JSON.stringify($('#calendar').selectpicker('val')),
        granularity_unit: $('#granularity').selectpicker('val'),   
        granularity_frequence: document.getElementById("granularity-value").value,
        mode: document.querySelector('input[name="check-mode"]:checked').value,
        pairwise_comparisons: document.querySelector('input[name="check-pairwise-comparisons"]:checked').value,
        correlations: JSON.stringify($('#correlation').selectpicker('val')),
        pca: document.getElementById("pca").checked  
    };
    console.log(data)
    $.post("receiver", data, receive_data);
    event.preventDefault();

};

function receive_data(data, status) {

    var data_json = JSON.parse(data);
    var lc_data;

    if (data_json.hasOwnProperty('line_chart')) {
        //create_line_chart(data_json["line_chart"]); 
    };

    if (data_json.hasOwnProperty('heat_map')) {
        create_heat_map(data_json["heat_map"]);
    };

    if (data_json.hasOwnProperty('line_chart_2')) {
        create_line_chart(data_json["line_chart_2"]);
    };


  
    activate_form();

    $('#collapse-visualization-settings-card').collapse('show');
    compress_expand_visualization("expand");
    $("#visualizaition-help-text-card-body").remove();
};

function validate_selection(field, length) {
    if ($('#' + field).val().length < length) {
        document.getElementById(field).setCustomValidity(' ');
        if (!document.getElementById("help-" + field)) {
            $("#div-" + field).append('<small id="help-' + field + '" class="text-muted">Select at least ' + length + ' </small>');
        };
    } else {
        document.getElementById(field).setCustomValidity('');
        if (document.getElementById("help-" + field)) {
            $("#help-" + field).remove();
        };
    }
};

function validate_check_box(field, length) {
    if ($("[name='check-" + field + "']:checked").length < length) {
        document.querySelectorAll("input[name=check-" + field + "]").forEach(box => {
            box.setCustomValidity(' ');
        });
        if (!document.getElementById("help-" + field)) {
            $("#div-" + field).append('<small id="help-' + field + '" class="text-muted">Check at least ' + length + ' </small>');
        };
    } else {
        document.querySelectorAll("input[name=check-" + field + "]").forEach(box => {
            box.setCustomValidity('');
        });
        if (document.getElementById("help-" + field)) {
            $("#help-" + field).remove();
        };
    };
};

/*
$('input[name="date-range"]').on('apply.daterangepicker', function (ev, picker) {
    console.log(picker.startDate.format('DD-MM-YYYY'));
    console.log(picker.endDate.format('DD-MM-YYYY'));
});
*/

function create_line_chart(lc_data) {
    var config = {
        responsive: true,
        scrollZoom: true
    };

    var layout = {
        xaxis: {
            //rangeslider: {}
            zeroline: false
        }, 
        yaxis: {
            title: {
                text: 'Flow [m<sup>3</sup>/h]',
            },
            zeroline: false
        },
        legend: {"orientation": "h"},
        margin: {
            //l: 50,
            //r: 50,
            //b: 0,
            t: 20,
            //pad: 4
          },
        yaxis2: {
            zeroline: false,
            title: 'Pressure [bar]',
            overlaying: 'y',
            side: 'right'
        },
    };

    var data = [];

    for (var sensor_id in lc_data) {
        if (!sensor_list.hasOwnProperty(sensor_id)) {
            continue;
        };
        var x = Object.keys(lc_data[sensor_id]);
        var y_aux = Object.values(lc_data[sensor_id]);
        y = []
        y_aux.forEach(row => {
            y.push(row['value'])
        });

        var trace = create_trace(sensor_id, x, y);
        data.push(trace);
    };
/*
    for (sensor in lc_data) {

        var x = Object.keys(lc_data[sensor]);
        var y = Object.values(lc_data[sensor]);

        var trace = create_trace(sensor, x, y);
        data.push(trace);
    };
*/
    $('#collapse-line_chart-section').collapse('show');
    $('#collapse-line_chart').collapse('show');
    Plotly.newPlot('line_chart', data, layout, config);  
};

function create_trace(name, x, y) {
    var trace;
    if(name == 3){
        trace = {
            name: name,
            x: x,
            y: y,
            type: 'scatter',
            yaxis: 'y2'
        };
    } else {
        trace = {
            name: name,
            x: x,
            y: y,
            type: 'scatter'
        };
    }

    return trace;
};


function create_heat_map(hm_data) {

    var config = {
        responsive: true
    };

    var layout = {
        margin: {
            //l: 50,
            //r: 50,
            //b: 0,
            t: 40,
            //pad: 4
          }
    };

    var data = [{
        z: [
            [-1, 0.5, -0.5],
            [0.3, 1, 0.8],
            [-0.8, 0, -0.7]
        ],
        x: ['1<br>(R, TLMT, F)', '2<br>(R, TLMT, F)', '3<br>(R, TLMT, F)'],
        y: ['1<br>(R, TLMT, F)', '2<br>(R, TLMT, F)', '3<br>(R, TLMT, F)'],
        colorscale: 'RdBu',
        //reversescale: true,
        zmin: -1,
        zmax: 1,
        type: 'heatmap',
        hoverongaps: false
    }];
    
    $('#collapse-heat_map-section').collapse('show');
    $('#collapse-heat_map').collapse('show');
    Plotly.newPlot('heat_map', data, layout, config);
};


function init_form(wme) {
    update_form(wme);
    create_date_range_picker("01/06/2017", "08/06/2017", "01/01/2017", "31/12/2017", 7);
    $('#calendar').selectpicker('refresh');
    $('#calendar').selectpicker('selectAll');
    $('#granularity').selectpicker('val', 'hours');
    document.getElementById("granularity-value").value = 1;
    document.getElementById("check-default").checked = true;
    document.getElementById("check-all-pairs").checked = true;
    $('#correlation-type').selectpicker('refresh');
    $('#correlation-type').selectpicker('selectAll');
    update_correlation();
    $('#correlation').selectpicker('selectAll');
    $("[name='check-pca'").prop("checked", true);
};


function create_date_range_picker(startDate, endDate, minDate, maxDate, maxSpan) {
    $(function () {
        $('input[name="date-range"]').daterangepicker({
            locale: {
                format: 'DD/MM/YYYY'
            },
            opens: 'center',
            "startDate": startDate,
            "endDate": endDate,
            "minDate": minDate,
            "maxDate": maxDate,
            "maxSpan": {
                "days": maxSpan
            }
        }, function (start, end, label) {
            //console.log("A new date selection was made: " + start.format('YYYY-MM-DD') + ' to ' + end.format('YYYY-MM-DD'));
        });
    });
};


function deactivate_form() {
    $('#run-btn').prop("disabled", true);
    $("#run-btn").html(
        `<span id="btn-spinner" class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Running...`
    );

    $('#source').prop("disabled", true);
    $('#source').selectpicker('refresh');
    $('#form-target input[type=checkbox]').prop("disabled", true);
    $('#sensor-type').prop("disabled", true);
    $('#sensor-type').selectpicker('refresh');
    $('#sensor-name').prop("disabled", true);
    $('#sensor-name').selectpicker('refresh');
    $('#calendar').prop("disabled", true);
    $('#calendar').selectpicker('refresh');
    $('#form-target input[type=text]').prop("disabled", true);
    $('#granularity').prop("disabled", true);
    $('#granularity').selectpicker('refresh');
    $('#form-target input[type=number]').prop("disabled", true);

    $('#form-target input[type=radio]').prop("disabled", true);
    $('#correlation-type').prop('disabled', true);
    $('#correlation-type').selectpicker('refresh');
    $('#correlation').prop('disabled', true);
    $('#correlation').selectpicker('refresh');

    $(':button[type=button]').prop("disabled", true);

    $('#form-visualization-settings input[type=checkbox]').prop("disabled", true);
    $('#vis-sensor-type').prop("disabled", true);
    $('#vis-sensor-type').selectpicker('refresh');
    $('#vis-sensor-name').prop("disabled", true);
    $('#vis-sensor-name').selectpicker('refresh');

};

function activate_form() {
    $('#source').prop("disabled", false);
    $('#source').selectpicker('refresh');
    $('#form-target input[type=checkbox]').prop("disabled", false);
    $('#sensor-type').prop("disabled", false);
    $('#sensor-type').selectpicker('refresh');
    $('#sensor-name').prop("disabled", false);
    $('#sensor-name').selectpicker('refresh');
    $('#calendar').prop("disabled", false);
    $('#calendar').selectpicker('refresh');
    $('#form-target input[type=text]').prop("disabled", false);
    $('#granularity').prop("disabled", false);
    $('#granularity').selectpicker('refresh');
    $('#form-target input[type=number]').prop("disabled", false);

    $('#form-target input[type=radio]').prop("disabled", false);
    $('#correlation-type').prop('disabled', false);
    $('#correlation-type').selectpicker('refresh');
    $('#correlation').prop('disabled', false);
    $('#correlation').selectpicker('refresh');

    if (!focus.real) {
        $("[value='real'][name='check-focus']").prop("disabled", true);
    }

    if (!focus.simulated) {
        $("[value='simulated'][name='check-focus']").prop("disabled", true);
    }

    if (!group.telemanagement) {
        $("[value='telemanagement'][name='check-sensor-group']").prop("disabled", true);
    }

    if (!group.telemetry) {
        $("[value='telemetry'][name='check-sensor-group']").prop("disabled", true);
    }

    $('#form-visualization-settings input[type=checkbox]').prop("disabled", false);
    $('#vis-sensor-type').prop("disabled", false);
    $('#vis-sensor-type').selectpicker('refresh');
    $('#vis-sensor-name').prop("disabled", false);
    $('#vis-sensor-name').selectpicker('refresh');

    $("#btn-spinner").remove();
    $("#run-btn").html(
        `Run Query`
    );
    $("#run-btn").prop("disabled", false);

    $(':button[type=button]').prop("disabled", false);
};