var wmes = 0;
var sensor_list = 0;
var selected_correlations = 0;
var hm_selected_data;
var selected_data = 0;

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

    document.getElementById("btn-redo-analysis-settings").addEventListener("click", function () {
        init_form($('#source').selectpicker('val'));
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            perform_manual_validation();
            form.checkValidity();
        };
        $('#collapse-target').collapse('show');
        $('#collapse-analysis').collapse('show');
    });

    document.getElementById("btn-collapse-visualization").addEventListener("click", function () {
        if ($('#btn-collapse-visualization').prop('name') == "not-active") {
            compress_expand_visualization("compress");
        } else {
            compress_expand_visualization("expand");
        };
    });

    document.getElementById("btn-swap-columns-corr").addEventListener("click", function () {
        create_heat_map("swap");
    });

    document.getElementById("btn-lock-columns-corr").addEventListener("click", function () {
        if(this.checked){
            document.getElementById("btn-lock-columns-corr-image").className = "fa fa-lock";
            document.getElementById("btn-lock-columns-corr-label").title = "Unlock Columns";
            create_heat_map("lock");
        } else {
            document.getElementById("btn-lock-columns-corr-image").className = "fa fa-lock-open";
            document.getElementById("btn-lock-columns-corr-label").title = "Lock Columns";
            create_heat_map("unlock");
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
        update_granularity_calendar();
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

    document.getElementById("corr-correlation").addEventListener("change", function () {
        create_heat_map("change-correlation");
    });

    document.getElementById("corr-order-by").addEventListener("change", function () {
        create_heat_map("change-order");
    });

    get_init_json();
};

function add_event_listener_collapsers(name, is_chart) {
    $('#collapse-' + name).on('hidden.bs.collapse', function () {
        $("#btn-collapse-" + name).html('<i class="fa fa-chevron-down"></i>');
        document.getElementById('btn-collapse-' + name).title = 'Show';
        if (is_chart) {
            Plotly.Plots.resize(name);
        };
    });
    $('#collapse-' + name).on('shown.bs.collapse', function () {
        $("#btn-collapse-" + name).html('<i class="fa fa-chevron-up"></i>');
        document.getElementById('btn-collapse-' + name).title = 'Hide';
        if ((name == "target") || (name == "analysis")) {
            compress_expand_visualization("compress")
        }
        if (is_chart) {
            Plotly.Plots.resize(name);
        };
    });
};

function compress_expand_visualization(type) {
    if (type == "compress") {
        $('#btn-collapse-visualization').prop('name', 'active');
        document.getElementById('row-form').className = "col-sm-4";
        document.getElementById('row-visualization').className = "col-sm-8";
        document.getElementById('btn-collapse-visualization').title = 'Expand';
        $("#btn-collapse-visualization").html('<i class="fa fa-expand"></i>');
    } else {
        $('#btn-collapse-visualization').prop('name', "not-active");
        document.getElementById('row-form').className = "col-sm-2";
        document.getElementById('row-visualization').className = "col-sm-10";
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
            var data_tokens = sensor["name"] + " " + sensor["focus"] + " " + sensor["group"] + " " + sensor["type"];
            var data_subtext = sensor["name"] + " • " + dim(sensor["focus"]) + " • " + dim(sensor["group"]) + " • " + dim(sensor["type"]);
            var option = '<option value="' + key + '" data-subtext="' + data_subtext + '" data-tokens="' + data_tokens + '">' + sensor["name_long"] + '</option>';
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

    update_granularity_calendar();
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

function create_correlation_option(value) {
    var subtext;
    var name;
    switch (value) {
        case "dcca":
            subtext = "Tuples";
            name = "DCCA";
            break;
        /*
            case "dcca-ln":
            subtext = "Tuples";
            name = "DCCA-ln";
            break;
        */
        case "pearson":
            subtext = "Temporal";
            name = "Pearson";
            break;
        case "kullback-leibler":
            subtext = "Temporal";
            name = "Kullback-Leibler";
            break;
    }
    return '<option data-subtext="' + subtext + '" data-tokens="Tuples" value="' + value + '">' + name + '</option>';
}

function update_correlation() {

    var selection = $('#correlation-type').val();
    var correlation_temporal = [
        ["dcca", "DCCA"]
        //["dcca-ln", "DCCA-ln"]
    ];
    var correlation_tuples = [
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

    selected_data = JSON.parse(data);

    console.log(selected_data)

    if (selected_data.hasOwnProperty('line_chart')) {
        create_line_chart(selected_data["line_chart"]);
    };

    if (selected_data.hasOwnProperty('heat_map')) {
        selected_correlations = Object.keys(selected_data["heat_map"])
        create_heat_map("create");
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

function create_line_chart(lc_data) {
    var config = {
        responsive: true,
        scrollZoom: true
    };

    var layout = {
        xaxis: {
            //rangeslider: {}
            zeroline: false,
        },
        yaxis: {
            title: {
                text: 'Flow [m<sup>3</sup>/h]',
            },
            zeroline: false,
            automargin: true
        },
        legend: {
            orientation: 'h',
            xanchor: 'center',
            x: 0.5
        },

        margin: {
            //l: 50,
            //r: 50,
            b: 0,
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

    $('#collapse-line_chart-section').collapse('show');
    $('#collapse-line_chart').collapse('show');
    Plotly.newPlot('line_chart', data, layout, config);
};

function create_trace(sensor_id, x, y) {
    var trace;
    var name = create_sensor_name(sensor_id);
    var sensor = sensor_list[sensor_id];

    if (sensor['type'] == 'pressure') {
        trace = {
            name: name,
            x: x,
            y: y,
            type: 'scatter',
            yaxis: 'y2',
            connectgaps: false
        };
    } else {
        trace = {
            name: name,
            x: x,
            y: y,
            type: 'scatter',
            connectgaps: false
        };
    };

    return trace;
};

function fix_x_y(x) {
    var x_aux = x.slice(0);
    x.forEach(function (part, index) {
        x_aux[index] = create_sensor_name(x[index])
    });
    return x_aux;
};

function create_heat_map(type) {

    var hm_variables;
    var showticklabels = false;
    var ticks = '';

    switch (type) {
        case "create":
            update_select_correlations_correlogram();
            hm_selected_data = selected_data['heat_map'][selected_correlations[0]];
            if(document.getElementById("btn-lock-columns-corr").checked){
                hm_variables = transform_heat_map_data("lock");
                showticklabels = true;
                ticks = 'outside';
                document.getElementById("btn-swap-columns-corr").disabled = true;
                $('#corr-order-by').prop('disabled', true);
                $('#corr-order-by').selectpicker('refresh');
            } else {
                hm_variables = transform_heat_map_data($('#corr-order-by').selectpicker('val'));
                $('#corr-order-by').prop('disabled', false);
                $('#corr-order-by').selectpicker('refresh');
                if (selected_data['pairwise_comparisons'] == "all pairs") {
                    document.getElementById("btn-swap-columns-corr").disabled = true;
                } else {
                    document.getElementById("btn-swap-columns-corr").disabled = false;
                };
            };

            break;
        case "swap":
            swap_heat_map_data();
            hm_variables = transform_heat_map_data($('#corr-order-by').selectpicker('val'));
            break;
        case "change-order":
            hm_variables = transform_heat_map_data($('#corr-order-by').selectpicker('val'));
            break;
        case "change-correlation":
            var selected_correlation = $('#corr-correlation').selectpicker('val');
            hm_selected_data = selected_data['heat_map'][selected_correlation];
            if(document.getElementById("btn-lock-columns-corr").checked){
                hm_variables = transform_heat_map_data("lock");
                showticklabels = true;
                ticks = 'outside';
            } else {
                hm_variables = transform_heat_map_data($('#corr-order-by').selectpicker('val'));
            }
            break;
        case "lock":
            hm_variables = transform_heat_map_data("lock");
            showticklabels = true;
            ticks = 'outside';
            document.getElementById("btn-swap-columns-corr").disabled = true;
            $('#corr-order-by').prop('disabled', true);
            $('#corr-order-by').selectpicker('refresh');
            break;
        case "unlock":
            var selected_correlation = $('#corr-correlation').selectpicker('val');
            hm_selected_data = selected_data['heat_map'][selected_correlation];
            hm_variables = transform_heat_map_data($('#corr-order-by').selectpicker('val'));
            $('#corr-order-by').prop('disabled', false);
            $('#corr-order-by').selectpicker('refresh');
            if (selected_data['pairwise_comparisons'] == "all pairs") {
                document.getElementById("btn-swap-columns-corr").disabled = true;
            } else {
                document.getElementById("btn-swap-columns-corr").disabled = false;
            };
            break;
    }

    var config = {
        responsive: true
    };

    var layout = {
        margin: {
            //l: 50,
            //r: 50,
            //b: 0,
            t: 20,
            //pad: 4
        },
        yaxis: {
            automargin: true,
            type: 'category'
        },
        xaxis: {
            automargin: true,
            showticklabels: showticklabels,
            ticks: ticks
        }
    };

    console.log(fix_x_y(hm_variables.y))

    var data = [{
        z: hm_variables.z,
        x: fix_x_y(hm_variables.x),
        y: fix_x_y(hm_variables.y),

        text: hm_variables.text,
        hovertemplate: hm_variables.hovertemplate,
        colorscale: hm_variables.colorscale,
        //reversescale: true,
        zmin: hm_variables.zmin,
        zmax: hm_variables.zmax,
        xgap: 3,
        ygap: 3,
        type: 'heatmap',
        //hoverongaps: false
    }];

    $('#collapse-heat_map-section').collapse('show');
    $('#collapse-heat_map').collapse('show');
    Plotly.newPlot('heat_map', data, layout, config);
};

function update_select_correlations_correlogram() {
    $('#corr-correlation').empty();
    $('#corr-correlation').prop("disabled", false);
    selected_correlations.forEach(key => {
        var option = create_correlation_option(key)
        $('#corr-correlation').append(option);
    });
    $('#corr-correlation').selectpicker('refresh');
    $('#corr-correlation').selectpicker('render');
}

function create_sensor_name(id) {

    var name = "";

    // todo (se calhar devia ser mais descritivo)

    if (wmes.infraquinta.hasOwnProperty(id)) {
        name = wmes.infraquinta[id]['name_long']
    } else {
        if (wmes.barreiro.hasOwnProperty(id)) {
            name = wmes.barreiro[id]['name_long']
        } else {
            name = wmes.beja[id]['name_long']
        };
    };
    return name;
}


function swap_heat_map_data() {

    var dic = hm_selected_data;

    var dic_keys = Object.keys(dic);
    var dic_aux_keys = [];

    dic[dic_keys[0]].forEach(element => {
        dic_aux_keys.push(element.id);
    });

    dic_aux_keys.sort(function (a, b) {
        return parseInt(a) - parseInt(b);
    });

    var dic_aux = {}

    dic_aux_keys.forEach(key => {
        var dic_aux_aux = []
        dic_keys.forEach(key1 => {
            var element1;
            dic[key1].forEach(element => {
                if (element.id == key) {
                    element1 = element;
                };
            });
            dic_aux_aux.push({
                'id': key1,
                'corr': element1.corr,
                'dist': element1.dist
            });
        });
        dic_aux[key] = dic_aux_aux;
    });

    hm_selected_data = dic_aux;

    console.log(dic_aux)

    //console.log(hm_selected_data);
}



function transform_heat_map_data_2() {

    var dic = {0: [{id:"2", dist: 3037.821, corr: -0.016},{id:"4", dist: 3037.821, corr: -0.016}], 
    1: [{id:"2", dist: 1702.6, corr: 0.01},{id:"4", dist: 3037.821, corr: -0.016}], 
    3: [{id: "2", dist: 0.406, corr: -0.957},{id:"4", dist: 3037.821, corr: -0.016}]};

    console.log(dic);
    var y = Object.keys(dic);
    y.sort(function (a, b) {
        return parseInt(a) - parseInt(b);
    });

    var x = [];

    dic[0].forEach(element => {
       x.push(element.id) 
    });

    var matrix_z = [];
    var matrix_text = [];

    y.forEach(ykey => {
        var matrix_z_row = [];
        var matrix_text_row = [];
        x.forEach(xkey => {   
            dic[parseInt(ykey)].forEach(yelement => {
                if(yelement.id == xkey){
                    var yelement_aux;
                    if(yelement.corr == 999999999){
                        yelement_aux = null;
                    } else {
                        yelement_aux = element.corr;
                    };
                    matrix_z_row.push(yelement_aux);
                    matrix_text_row.push('X: ' + create_sensor_name(yelement.id) + '<br>Dist: ' + yelement.dist + ' m');
                };
            });     
        });
        matrix_z.push(matrix_z_row);
        matrix_text.push(matrix_text_row);     
    });

    console.log(matrix_z)
    console.log(matrix_text)
    
}

//transform_heat_map_data_2()


function transform_heat_map_data(sortby) {

    var dic = hm_selected_data;
    console.log(dic);

    var matrix_z = [];
    var matrix_text = [];
    var y = [];
    var x = [];
    var x_aux = [];

    if(sortby == "lock"){
        y = Object.keys(dic);
        y.sort(function (a, b) {
            return parseInt(a) - parseInt(b);
        });
    
        dic[Object.keys(dic)[0]].forEach(element => {
           x.push(element.id);
        });

        x.sort(function (a, b) {
            return parseInt(a) - parseInt(b);
        });
    
        y.forEach(ykey => {
            var matrix_z_row = [];
            var matrix_text_row = [];
            x.forEach(xkey => {   
                dic[parseInt(ykey)].forEach(yelement => {
                    if(yelement.id == xkey){
                        var yelement_aux;
                        if(yelement.corr == 999999999){
                            yelement_aux = null;
                        } else {
                            yelement_aux = yelement.corr;
                        };
                        matrix_z_row.push(yelement_aux);
                        matrix_text_row.push('X: ' + create_sensor_name(yelement.id) + '<br>Dist: ' + yelement.dist + ' m');
                    };
                });     
            });
            matrix_z.push(matrix_z_row);
            matrix_text.push(matrix_text_row);     
        });

        x_aux = x;

    } else {
    
        y = Object.keys(dic);
        y.sort(function (a, b) {
            return parseInt(a) - parseInt(b);
        });
    
        dic[y[0]].forEach(element => {
            x.push(element.id)
        });
    
        for (var i = 0; i < x.length; i++) {
            x_aux.push(i);
        }
    
        for (var key in dic) {
            if (!dic.hasOwnProperty(key)) {
                continue;
            };
    
            var sensor = dic[key];
    
            if (sortby == "distance") {
                sensor.sort(function (a, b) {
                    return a.dist - b.dist;
                });
            } else {
                sensor.sort(function (a, b) {
                    return a.corr - b.corr;
                });
            }
    
            var matrix_z_row = []
            var matrix_text_row = []
    
            // TODO (Não a correlação do sensor com ele próprio)
    
            sensor.forEach(element => {
                var element_aux;
                if(element.corr == 999999999){
                    element_aux = null;
                } else {
                    element_aux = element.corr;
                };
                matrix_z_row.push(element_aux);
                matrix_text_row.push('X: ' + create_sensor_name(element.id) + '<br>Dist: ' + element.dist + ' m')
            });
    
            matrix_z.push(matrix_z_row);
            matrix_text.push(matrix_text_row);
        };

    }

    console.log(x_aux)
    console.log(y)
    console.log(matrix_z)

    var zmin = -1;
    var zmax = 1;
    var colorscale = 'RdBu';
    var array_z = [];
    var hovertemplate = 'Y: %{y}<extra></extra>' + '<br>%{text}' + '<br>Corr: %{z}';

    if ($('#corr-correlation').selectpicker('val') == "kullback-leibler"){
        matrix_z.forEach(row_z => {
            row_z.forEach(element_z => {
                array_z.push(element_z);
            });    
        });
       zmin = Math.min.apply(null, array_z);
       zmax = Math.max.apply(null, array_z);
       colorscale = 'Greens';
       hovertemplate = 'Y: %{y}<extra></extra>' + '<br>%{text}' + '<br>Corr: %{z} bits';
    };

    return {
        'x': x_aux,
        'y': y,
        'z': matrix_z,
        'text': matrix_text,
        'zmin': zmin,
        'zmax': zmax,
        'colorscale': colorscale,
        'hovertemplate': hovertemplate
    }
}


function init_form(wme) {
    update_form(wme);
    $('#calendar').selectpicker('refresh');
    $('#calendar').selectpicker('selectAll');
    document.getElementById("check-default").checked = true;
    document.getElementById("check-all-pairs").checked = true;
    $('#correlation-type').selectpicker('refresh');
    $('#correlation-type').selectpicker('selectAll');
    update_correlation();
    $('#correlation').selectpicker('selectAll');
    $("[name='check-pca']").prop("checked", true);
};


function create_date_range_picker(startDate, endDate, minDate, maxDate, maxSpan) {
    if (maxSpan > 0) {
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
            });
        });
    } else {
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
            });
        });
    };
};

function get_min_max_date(type, dates_string) {
    var dates = []
    dates_string.forEach(date_string => {
        dates.push(convert_date(date_string));
    });
    var new_date;
    if (type == "max") {
        new_date = new Date(Math.max.apply(null, dates));
    } else {
        new_date = new Date(Math.min.apply(null, dates));
    }
    new_date = new Intl.DateTimeFormat().format(new_date);
    return new_date;
}

function convert_date(date) {
    var res = date.split("/");
    var date_aux = Date.parse(parseInt(res[2]) + "/" + parseInt(res[1]) + "/" + parseInt(res[0]));
    return date_aux;
}

function add_days(date, days) {
    var result = new Date(date);
    result.setDate(result.getDate() + days);
    return result;
}

function calculate_day_difference(date1, date2) {
    const diffTime = Math.abs(date2 - date1);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    return diffDays;
}

function get_correct_dates(current_start_date, current_end_date, start_date, end_date, maxSpan) {
    var correct_start;
    var correct_end;
    if ((current_start_date < start_date) || (current_start_date > end_date)) {
        correct_start = start_date;
    } else {
        correct_start = current_start_date;
    }
    if ((current_end_date > end_date) || (current_end_date < start_date)) {
        correct_end = end_date;
    } else {
        correct_end = current_end_date;
    }
    if ((maxSpan > 0) && (calculate_day_difference(correct_start, correct_end) > maxSpan)) {
        correct_end = add_days(correct_start, maxSpan);
    }
    correct_start = new Intl.DateTimeFormat().format(correct_start);
    correct_end = new Intl.DateTimeFormat().format(correct_end);
    return [correct_start, correct_end];
}

function update_granularity_calendar() {

    var granularity_units = [];
    var date_range_max = [];
    var date_range_min = [];
    var maxSpan = 0;

    var selected_sensors = $('#sensor-name').selectpicker('val');

    if (selected_sensors.length > 0) {

        if (sensor_list[selected_sensors[0]] != undefined) {

            $("#date-range").prop("disabled", false);
            $("#granularity").prop("disabled", false);
            $('#granularity').selectpicker('refresh');
            $("#granularity-value").prop("disabled", false);

            selected_sensors.forEach(key => {
                if ((sensor_list[key].min_date == "nd") || (sensor_list[key].max_date == "nd")) {
                    maxSpan = 6;
                } else {
                    date_range_min.push(sensor_list[key].min_date);
                    date_range_max.push(sensor_list[key].max_date);
                }
                granularity_units.push(parseInt(sensor_list[key]['granularity_unit']));
            });

            if ((date_range_min.length == 0) || (date_range_max.length == 0)) {
                var min_date_range = $('#date-range').data('daterangepicker').minDate.format('DD/MM/YYYY');
                var max_date_range = $('#date-range').data('daterangepicker').maxDate.format('DD/MM/YYYY');
                var current_min_date = $('#date-range').data('daterangepicker').startDate.format('DD/MM/YYYY');
                var current_max_date = $('#date-range').data('daterangepicker').endDate.format('DD/MM/YYYY');
                var correct_dates = get_correct_dates(convert_date(current_min_date), convert_date(current_max_date),
                    convert_date(min_date_range), convert_date(max_date_range), maxSpan);
                create_date_range_picker(correct_dates[0], correct_dates[1], min_date_range, max_date_range, maxSpan);
            } else {
                var min_date_range = get_min_max_date("max", date_range_min);
                var max_date_range = get_min_max_date("min", date_range_max);
                if ($('#date-range').data('daterangepicker') === undefined) {
                    var correct_dates = get_correct_dates(convert_date(min_date_range), convert_date(max_date_range),
                        convert_date(min_date_range), convert_date(max_date_range), maxSpan);
                    create_date_range_picker(correct_dates[0], correct_dates[1], min_date_range, max_date_range, maxSpan);
                } else {
                    var current_min_date = $('#date-range').data('daterangepicker').startDate.format('DD/MM/YYYY');
                    var current_max_date = $('#date-range').data('daterangepicker').endDate.format('DD/MM/YYYY');
                    var correct_dates = get_correct_dates(convert_date(current_min_date), convert_date(current_max_date),
                        convert_date(min_date_range), convert_date(max_date_range), maxSpan);
                    create_date_range_picker(correct_dates[0], correct_dates[1], min_date_range, max_date_range, maxSpan);
                };
            };

            var max_granularity_unit = Math.max.apply(null, granularity_units);

            selected_granularity_unit = parseInt($('#granularity').selectpicker('val'));
            if ((selected_granularity_unit < max_granularity_unit) || !selected_granularity_unit) {
                $('#granularity').selectpicker('val', max_granularity_unit);
            };

            switch (max_granularity_unit) {
                case 0:
                    $("#granularity option").prop("disabled", false);
                    $("#granularity option[value='1']").prop("disabled", false);
                    $("#granularity option[value='2']").prop("disabled", false);
                    $("#granularity option[value='3']").prop("disabled", false);
                    break;
                case 1:
                    $("#granularity option[value='0']").prop("disabled", true);
                    $("#granularity option[value='1']").prop("disabled", false);
                    $("#granularity option[value='2']").prop("disabled", false);
                    $("#granularity option[value='3']").prop("disabled", false);
                    break;
                case 2:
                    $("#granularity option[value='0']").prop("disabled", true);
                    $("#granularity option[value='1']").prop("disabled", true);
                    $("#granularity option[value='2']").prop("disabled", false);
                    $("#granularity option[value='3']").prop("disabled", false);
                    break;
                case 3:
                    $("#granularity option[value='0']").prop("disabled", true);
                    $("#granularity option[value='1']").prop("disabled", true);
                    $("#granularity option[value='2']").prop("disabled", true);
                    $("#granularity option[value='3']").prop("disabled", false);
                    break;
            };
        };

    } else {
        $("#granularity").prop("disabled", true);
        $("#granularity-value").prop("disabled", true);
        $("#date-range").prop("disabled", true);
    };
    $('#granularity').selectpicker('refresh');
    $('#granularity').selectpicker('render');
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

    $(':button[btn-type=collapse]').prop("disabled", true);
    $(':button[btn-type=redo]').prop("disabled", true);

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

    if (focus.real == false) {
        $("[value='real'][name='check-focus']").prop("disabled", true);
    }

    if (focus.simulated == false) {
        $("[value='simulated'][name='check-focus']").prop("disabled", true);
    }

    if (group.telemanagement == false) {
        $("[value='telemanagement'][name='check-sensor-group']").prop("disabled", true);
    }

    if (group.telemetry == false) {
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

    $(':button[btn-type=collapse]').prop("disabled", false);
    $(':button[btn-type=redo]').prop("disabled", false);

};